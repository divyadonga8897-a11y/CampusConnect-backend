import os
from sqlalchemy.orm import Session
from app.models.models import (
    College, Department, Course, FeeStructure, Scholarship, Facility, PlacementStatistics, Alumni
)

# Optional third-party imports
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

try:
    from groq import Groq
except ImportError:
    Groq = None


class RagService:
    def __init__(self):
        from app.core.config import settings
        self.openai_key = settings.OPENAI_API_KEY or os.getenv("OPENAI_API_KEY", "")
        self.groq_key = settings.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
        self.pinecone_key = settings.PINECONE_API_KEY or os.getenv("PINECONE_API_KEY", "")
        self.pinecone_index = getattr(settings, 'PINECONE_INDEX_NAME', '') or os.getenv("PINECONE_INDEX_NAME", "campusconnect-ai")
        
        self.openai_client = OpenAI(api_key=self.openai_key) if (OpenAI and self.openai_key) else None
        self.groq_client = Groq(api_key=self.groq_key) if (Groq and self.groq_key) else None

    def query_assistant(self, prompt: str, history: list, db: Session) -> str:
        # Check if RAG is operational via external credentials
        if (self.groq_client or self.openai_client) and self.pinecone_key:
            res = self._query_rag_pipeline(prompt, history)
            if res:
                return res
        return self._query_local_semantic_router(prompt, db)

    def _query_rag_pipeline(self, prompt: str, history: list) -> str:
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_key)
            index = pc.Index(self.pinecone_index)
            
            # 1. Generate query embedding
            embeddings = pc.inference.embed(
                model="multilingual-e5-large",
                inputs=[prompt],
                parameters={"input_type": "query", "truncate": "END"}
            )
            query_vector = embeddings[0].values
            
            # 2. Query Pinecone
            results = index.query(
                vector=query_vector,
                top_k=5,
                include_metadata=True
            )
            
            # 3. Retrieve relevant context
            context_parts = []
            sources = set()
            for match in results.get("matches", []):
                if match.score >= 0.5:  # Relevance threshold
                    meta = match.get("metadata", {})
                    text = meta.get("text", "")
                    filename = meta.get("filename", "")
                    if text:
                        context_parts.append(text)
                    if filename:
                        sources.add(filename)
            
            if not context_parts:
                return ""  # Trigger local semantic fallback
            
            context = "\n---\n".join(context_parts)
            sources_list = list(sources)
            sources_suffix = f"\n\n**Sources:** {', '.join(sources_list)}" if sources_list else ""
            
            system_instruction = (
                f"You are the CampusConnect AI Assistant for Sri Satya Institute of Engineering and Technology. "
                f"Answer the user's question accurately based ONLY on the following context. If the answer cannot be found in the context, politely state that you do not know. "
                f"Be professional, clear, and structured in your response.\n\n"
                f"Context:\n{context}"
            )
            
            messages = [{"role": "system", "content": system_instruction}]
            for h in history:
                messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
            messages.append({"role": "user", "content": prompt})
            
            if self.groq_client:
                completion = self.groq_client.chat.completions.create(
                    model="llama-3.1-8b-instant",
                    messages=messages,
                    temperature=0.7,
                )
                return completion.choices[0].message.content + sources_suffix
            elif self.openai_client:
                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                )
                return completion.choices[0].message.content + sources_suffix
                
        except Exception as e:
            print(f"[RAG] Error querying RAG pipeline: {e}")
            return ""

        return ""

    def process_and_index_document(self, doc_id: str, file_path: str, filename: str, category: str, file_type: str, db: Session):
        """
        Background task: extracts text, recursively chunks it, generates embeddings, and upserts to Pinecone index.
        """
        try:
            print(f"[RAG-Index] Starting background indexing for document {doc_id} ({filename})...")
            
            # 1. Extract text
            text = ""
            if file_type == "pdf":
                import pypdf
                reader = pypdf.PdfReader(file_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            elif file_type == "docx":
                import docx
                doc = docx.Document(file_path)
                text = "\n".join([para.text for para in doc.paragraphs])
            else: # txt or md
                with open(file_path, "r", encoding="utf-8") as f:
                    text = f.read()
            
            if not text.strip():
                raise ValueError("Extracted text is empty")
            
            # 2. Recursive Chunking
            chunks = self._split_text_recursive(text, max_chunk_size=1000, overlap=200)
            print(f"[RAG-Index] Document split into {len(chunks)} chunks.")
            
            # 3. Connect to Pinecone and Upsert
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_key)
            index = pc.Index(self.pinecone_index)
            
            # Embed chunks in batches of 32
            batch_size = 32
            vectors = []
            
            for i in range(0, len(chunks), batch_size):
                batch_chunks = chunks[i:i + batch_size]
                embeddings_res = pc.inference.embed(
                    model="multilingual-e5-large",
                    inputs=batch_chunks,
                    parameters={"input_type": "passage", "truncate": "END"}
                )
                
                for idx, embedding in enumerate(embeddings_res):
                    chunk_idx = i + idx
                    vector_id = f"{doc_id}_{chunk_idx}"
                    vectors.append({
                        "id": vector_id,
                        "values": embedding.values,
                        "metadata": {
                            "document_id": doc_id,
                            "filename": filename,
                            "category": category,
                            "text": batch_chunks[idx]
                        }
                    })
            
            # Upsert vectors to Pinecone
            print(f"[RAG-Index] Upserting vectors to Pinecone...")
            index.upsert(vectors=vectors)
            
            # 4. Update Database
            from app.models.models import KnowledgeDocument
            import datetime
            doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
            if doc:
                doc.status = "Indexed"
                doc.chunk_count = len(chunks)
                doc.indexed_status = True
                doc.updated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                db.commit()
                print(f"[RAG-Index] Document {doc_id} successfully indexed!")
                
        except Exception as e:
            print(f"[RAG-Index] Indexing failed for document {doc_id}. Error: {e}")
            from app.models.models import KnowledgeDocument
            import datetime
            doc = db.query(KnowledgeDocument).filter(KnowledgeDocument.id == doc_id).first()
            if doc:
                doc.status = "Failed"
                doc.updated_at = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
                db.commit()

    def delete_document_vectors(self, doc_id: str):
        """
        Delete all vector embeddings for a given document from the Pinecone index.
        """
        try:
            from pinecone import Pinecone
            pc = Pinecone(api_key=self.pinecone_key)
            index = pc.Index(self.pinecone_index)
            print(f"[RAG-Index] Deleting vectors for document {doc_id} from Pinecone...")
            index.delete(filter={"document_id": doc_id})
        except Exception as e:
            print(f"[RAG-Index] Error deleting vectors for document {doc_id} from Pinecone: {e}")

    def _split_text_recursive(self, text: str, max_chunk_size: int = 1000, overlap: int = 200) -> list:
        separators = ["\n\n", "\n", " ", ""]
        chunks = []
        
        def split_helper(current_text: str, current_separator_idx: int):
            if len(current_text) <= max_chunk_size:
                chunks.append(current_text.strip())
                return
            
            if current_separator_idx >= len(separators):
                # Hard slice if no separators left
                start = 0
                while start < len(current_text):
                    chunks.append(current_text[start:start + max_chunk_size].strip())
                    start += max_chunk_size - overlap
                return
            
            sep = separators[current_separator_idx]
            parts = current_text.split(sep) if sep else list(current_text)
            current_chunk = ""
            
            for part in parts:
                if len(current_chunk) + len(part) + (len(sep) if current_chunk else 0) <= max_chunk_size:
                    if current_chunk:
                        current_chunk += sep + part
                    else:
                        current_chunk = part
                else:
                    if current_chunk:
                        chunks.append(current_chunk.strip())
                        overlap_len = min(overlap, len(current_chunk))
                        current_chunk = current_chunk[-overlap_len:] + sep + part
                    else:
                        split_helper(part, current_separator_idx + 1)
            
            if current_chunk:
                chunks.append(current_chunk.strip())
        
        split_helper(text, 0)
        return [c for c in chunks if c]


    def _query_local_semantic_router(self, prompt: str, db: Session) -> str:
        query = prompt.lower()

        # 1. Fees Intent
        if any(w in query for w in ["fee", "fees", "cost", "charge", "tuition"]):
            fees = db.query(FeeStructure).all()
            if not fees:
                return "The college fee structure varies between B.Tech departments. Generally, annual tuition fees are around 75,000 to 90,000 INR. Optional hostel charges are 55,000 INR annually."
            
            fee_lines = []
            for f in fees:
                fee_lines.append(f"- **{f.course_id.replace('b-tech-', '').upper()}** ({f.fee_type}): Tuition Fee: {f.tuition_fee} INR/Yr | Hostel: {f.hostel_fee} INR/Yr")
            return (
                "Here is the fee structure for B.Tech programs at Sri Satya Institute of Engineering and Technology (A.Y. 2024-25):\n\n"
                + "\n".join(fee_lines) +
                "\n\n*Note: Transport fees are optional (18,000 INR/year). Standard examination and lab fees are charged at the beginning of semesters.*"
            )

        # 2. Scholarship Intent
        if any(w in query for w in ["scholarship", "scholarships", "concession", "reimbursement", "financial aid"]):
            schol = db.query(Scholarship).all()
            if not schol:
                return "Sri Satya Institute offers Merit Excellence Scholarships (up to 50% tuition waiver for 90%+ marks) and SC/ST Government fee reimbursements. Please contact the administrative office for details."
            
            sch_lines = []
            for s in schol:
                sch_lines.append(f"- **{s.title}**: Benefits include {', '.join(s.benefits)}. Eligibility: {', '.join(s.eligibility)}.")
            return (
                "Sri Satya Institute supports students through various financial aid programs:\n\n"
                + "\n".join(sch_lines) +
                "\n\n*You can apply for these during admission counseling by submitting caste, income, or 10+2 mark sheets.*"
            )

        # 3. Placements & Recruiters Intent
        if any(w in query for w in ["placement", "placements", "recruit", "recruiter", "recruiters", "salary", "package", "lpa"]):
            stats = db.query(PlacementStatistics).order_by(PlacementStatistics.year.desc()).first()
            alumni_count = db.query(Alumni).count()
            if stats:
                return (
                    f"Sri Satya Institute has an excellent placement record. In the recent **{stats.year} graduating batch**:\n\n"
                    f"- **Placement Percentage**: {stats.placement_percentage}%\n"
                    f"- **Highest Package Offered**: {stats.highest_package}\n"
                    f"- **Average Package**: {stats.average_package}\n"
                    f"- **Participating Companies**: {stats.companies_count}+\n\n"
                    f"Our graduates are placed at leading companies like TCS, Wipro, Infosys, Accenture, Amazon, and Qualcomm."
                )
            return "SSIET maintains a 90%+ placement rate. Our highest package reaches 14.5 LPA, with an average package of 5.1 LPA. Partner recruiters include TCS, Wipro, Infosys, and Tech Mahindra."

        # 4. Department / Course details
        if any(w in query for w in ["course", "courses", "department", "departments", "programs", "b.tech", "cse", "aids", "ece", "civil", "mech"]):
            depts = db.query(Department).all()
            dept_lines = [f"- **{d.department_name} ({d.short_name})** - HOD: {d.head_of_department}." for d in depts]
            return (
                "Sri Satya Institute of Engineering and Technology offers 5 specialized B.Tech programs:\n\n"
                + "\n".join(dept_lines) +
                "\n\nEach course is 4 years (8 semesters) in duration and requires 10+2 / intermediate MPC stream eligibility with EAMCET/JEE ranks."
            )

        # 5. Hostel Intent
        if any(w in query for w in ["hostel", "hostels", "mess", "dining", "room", "accommodation"]):
            return (
                "Sri Satya Institute provides separate residential hostels for boys and girls inside the college boundary walls:\n\n"
                "- **Accommodations**: Double & Triple sharing rooms fully furnished with tables, cupboards, and bedding.\n"
                "- **Food & Dining**: Clean dining halls serving nutritious vegetarian and non-vegetarian food mapped by student mess committees.\n"
                "- **Security**: 24/7 gate security, biometric logs, wardens resident on-site, and full CCTV coverage.\n"
                "- **Amenities**: High-speed campus Wi-Fi, late-hour reading rooms, and indoor sports zones."
            )

        # 6. Default Fallback Response
        college = db.query(College).first()
        college_name = college.name if college else "Sri Satya Institute of Engineering and Technology"
        return (
            f"Hello! I am the **CampusConnect AI Assistant** for {college_name}.\n\n"
            "I can help you explore:\n"
            "- 📚 **B.Tech Programs** & Engineering Departments\n"
            "- 💰 **Fee Structures** & Annual Tuition/Hostel costs\n"
            "- 🎓 **Scholarships** & Government Fee Reimbursements\n"
            "- 🏢 **Campus Facilities**, AI labs, and central libraries\n"
            "- 💼 **Placement Records** & Recruiting Partners\n"
            "- 🏠 **Hostel Life** and dining facilities\n\n"
            "What would you like to explore today? Try asking: *'What are the fees for B.Tech CSE?'* or *'What is the highest package offered?'*"
        )
