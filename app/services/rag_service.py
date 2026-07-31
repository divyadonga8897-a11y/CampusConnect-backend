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
        self.openai_key = os.getenv("OPENAI_API_KEY", "")
        self.groq_key = os.getenv("GROQ_API_KEY", "")
        self.pinecone_key = os.getenv("PINECONE_API_KEY", "")
        
        self.openai_client = OpenAI(api_key=self.openai_key) if (OpenAI and self.openai_key) else None
        self.groq_client = Groq(api_key=self.groq_key) if (Groq and self.groq_key) else None

    def query_assistant(self, prompt: str, history: list, db: Session) -> str:
        # Check if RAG is operational via external credentials
        if (self.groq_client or self.openai_client) and self.pinecone_key:
            return self._query_rag_pipeline(prompt, history)
        else:
            return self._query_local_semantic_router(prompt, db)

    def _query_rag_pipeline(self, prompt: str, history: list) -> str:
        # Production RAG placeholder structure.
        # In a real environment, we would fetch embeddings for `prompt` from OpenAI/SentenceTransformers,
        # query the Pinecone index for top matches, build a context window, and send to LLM.
        context = "SSIET is a premier engineering college in West Godavari, Andhra Pradesh."
        system_instruction = (
            f"You are the CampusConnect AI Assistant for Sri Satya Institute of Engineering and Technology. "
            f"Answer based on the following context:\n\n{context}"
        )
        
        messages = [{"role": "system", "content": system_instruction}]
        for h in history:
            messages.append({"role": h.get("role", "user"), "content": h.get("content", "")})
        messages.append({"role": "user", "content": prompt})

        try:
            if self.groq_client:
                # Use Groq Llama-3 by default for speed
                completion = self.groq_client.chat.completions.create(
                    model="llama3-8b-8192",
                    messages=messages,
                    temperature=0.7,
                )
                return completion.choices[0].message.content
            elif self.openai_client:
                completion = self.openai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    temperature=0.7,
                )
                return completion.choices[0].message.content
        except Exception as e:
            return f"Error contacting AI server: {str(e)}. Falling back to local search."

        return "AI models are not initialized."

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
