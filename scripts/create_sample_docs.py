"""Generate three realistic synthetic legal PDFs for development and testing.

Run directly:
    python scripts/create_sample_docs.py

Or imported as a module (used by test fixtures):
    from scripts.create_sample_docs import create_all_sample_docs
"""
import sys
from pathlib import Path

# Allow running from project root or from scripts/
sys.path.insert(0, str(Path(__file__).parent.parent))

from fpdf import FPDF  # fpdf2

from config import RAW_DIR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pdf(title: str) -> FPDF:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 12, title, align="C", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)
    pdf.set_font("Helvetica", size=11)
    return pdf


def _section(pdf: FPDF, heading: str, body: str) -> None:
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 9, heading, new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", size=11)
    pdf.multi_cell(0, 7, body.strip())
    pdf.ln(3)


# ---------------------------------------------------------------------------
# Document 1 - Employment Contract
# ---------------------------------------------------------------------------

def create_employment_contract() -> Path:
    out = RAW_DIR / "employment_contract.pdf"
    pdf = _pdf("EMPLOYMENT AGREEMENT")

    _section(pdf, "PARTIES", (
        "This Employment Agreement ('Agreement') is entered into as of January 1, 2024 "
        "('Effective Date'), by and between Acme Corporation, a Delaware corporation "
        "('Employer'), and Jane Smith, an individual residing at 42 Maple Street, "
        "Springfield, IL 62701 ('Employee')."
    ))

    _section(pdf, "SECTION 1 - POSITION AND DUTIES", (
        "1.1  Position. Employer hereby employs Employee as Senior Software Engineer. "
        "Employee shall report directly to the Chief Technology Officer.\n\n"
        "1.2  Duties. Employee shall perform all duties customarily associated with the "
        "position of Senior Software Engineer, including designing and implementing "
        "scalable software systems, conducting code reviews, and mentoring junior engineers.\n\n"
        "1.3  Full-Time Commitment. Employee agrees to devote substantially all of "
        "Employee's working time, attention, and efforts to the performance of duties "
        "under this Agreement."
    ))

    _section(pdf, "SECTION 2 - COMPENSATION AND BENEFITS", (
        "2.1  Base Salary. Employer shall pay Employee an annual base salary of "
        "One Hundred Twenty Thousand Dollars ($120,000), payable in accordance with "
        "Employer's standard payroll schedule (bi-weekly).\n\n"
        "2.2  Performance Bonus. Employee shall be eligible for an annual performance "
        "bonus of up to fifteen percent (15%) of base salary, based on achievement of "
        "mutually agreed-upon objectives reviewed each calendar quarter.\n\n"
        "2.3  Health Insurance. Employer shall provide Employee with health, dental, and "
        "vision insurance coverage under the Employer's group health plan, with premiums "
        "seventy percent (70%) employer-paid.\n\n"
        "2.4  Paid Time Off. Employee shall accrue twenty (20) days of paid time off per "
        "calendar year, accruing at 1.67 days per month, with a maximum carryover of "
        "ten (10) days into the following year.\n\n"
        "2.5  401(k) Plan. Employee is eligible to participate in Employer's 401(k) "
        "plan with an employer match of up to four percent (4%) of Employee's "
        "annual base salary."
    ))

    _section(pdf, "SECTION 3 - TERM AND TERMINATION", (
        "3.1  At-Will Employment. Employment under this Agreement is at-will, meaning "
        "either party may terminate the employment relationship at any time, with or "
        "without cause.\n\n"
        "3.2  Notice Period. Either party wishing to terminate this Agreement shall "
        "provide thirty (30) days written notice to the other party. Employer may, "
        "at its sole discretion, elect to pay Employee thirty (30) days base salary "
        "in lieu of notice.\n\n"
        "3.3  Severance. If Employer terminates Employee without cause after twelve (12) "
        "months of continuous employment, Employee shall receive a severance payment "
        "equal to three (3) months of base salary."
    ))

    _section(pdf, "SECTION 4 - CONFIDENTIALITY", (
        "4.1  Confidential Information. During and after the term of employment, "
        "Employee shall keep confidential all non-public information relating to "
        "Employer's business, including but not limited to trade secrets, client lists, "
        "financial data, source code, and product roadmaps ('Confidential Information').\n\n"
        "4.2  Return of Materials. Upon termination, Employee shall promptly return all "
        "Confidential Information and Employer property."
    ))

    _section(pdf, "SECTION 5 - NON-COMPETE AND NON-SOLICITATION", (
        "5.1  Non-Compete. For a period of twelve (12) months following termination, "
        "Employee shall not engage in or be employed by any business that directly "
        "competes with Employer's primary line of business within the United States.\n\n"
        "5.2  Non-Solicitation of Employees. For a period of twelve (12) months "
        "following termination, Employee shall not solicit, recruit, or induce any "
        "Employer employee to terminate their employment.\n\n"
        "5.3  Non-Solicitation of Clients. For a period of twelve (12) months following "
        "termination, Employee shall not solicit or attempt to obtain business from any "
        "client of Employer with whom Employee had material contact during employment."
    ))

    _section(pdf, "SECTION 6 - INTELLECTUAL PROPERTY", (
        "6.1  Work Product. All inventions, designs, works of authorship, and other "
        "intellectual property created by Employee within the scope of employment shall "
        "be the sole and exclusive property of Employer ('Work Product').\n\n"
        "6.2  Assignment. Employee hereby irrevocably assigns all right, title, and "
        "interest in and to the Work Product to Employer."
    ))

    _section(pdf, "SECTION 7 - GOVERNING LAW", (
        "This Agreement shall be governed by and construed in accordance with the laws "
        "of the State of Delaware, without regard to its conflict of laws principles. "
        "Any dispute arising under this Agreement shall be resolved by binding "
        "arbitration under the rules of the American Arbitration Association in "
        "Wilmington, Delaware."
    ))

    _section(pdf, "SIGNATURES", (
        "IN WITNESS WHEREOF, the parties have executed this Agreement as of the "
        "Effective Date.\n\n"
        "Acme Corporation\n"
        "By: Robert Johnson, CEO\n"
        "Date: January 1, 2024\n\n"
        "Employee\n"
        "Jane Smith\n"
        "Date: January 1, 2024"
    ))

    pdf.output(str(out))
    print(f"  Created: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Document 2 - Non-Disclosure Agreement
# ---------------------------------------------------------------------------

def create_nda() -> Path:
    out = RAW_DIR / "non_disclosure_agreement.pdf"
    pdf = _pdf("MUTUAL NON-DISCLOSURE AGREEMENT")

    _section(pdf, "PARTIES", (
        "This Mutual Non-Disclosure Agreement ('Agreement') is entered into as of "
        "March 15, 2024 ('Effective Date'), by and between TechCorp Inc, a California "
        "corporation ('Disclosing Party A'), and DataVault LLC, a New York limited "
        "liability company ('Disclosing Party B'). Each party may act as both disclosing "
        "and receiving party under this Agreement."
    ))

    _section(pdf, "SECTION 1 - DEFINITION OF CONFIDENTIAL INFORMATION", (
        "1.1  'Confidential Information' means any non-public information disclosed by "
        "one party ('Disclosing Party') to the other party ('Receiving Party'), either "
        "directly or indirectly, in writing, orally, or by inspection of tangible "
        "objects, that is designated as confidential or that reasonably should be "
        "understood to be confidential given the nature of the information and "
        "circumstances of disclosure.\n\n"
        "1.2  Confidential Information includes, without limitation: technical data, "
        "trade secrets, know-how, research, product plans, products, services, customer "
        "lists, markets, software, developments, inventions, processes, formulas, "
        "technology, designs, drawings, engineering, hardware configuration information, "
        "marketing data, financial information, and business plans."
    ))

    _section(pdf, "SECTION 2 - OBLIGATIONS OF RECEIVING PARTY", (
        "2.1  Non-Disclosure. The Receiving Party agrees to hold the Disclosing Party's "
        "Confidential Information in strict confidence and not to disclose such "
        "information to any third party without the prior written consent of the "
        "Disclosing Party.\n\n"
        "2.2  Limited Use. The Receiving Party shall use Confidential Information solely "
        "for the purpose of evaluating a potential business relationship between the "
        "parties ('Permitted Purpose') and for no other purpose.\n\n"
        "2.3  Standard of Care. The Receiving Party shall protect the Disclosing Party's "
        "Confidential Information using the same degree of care it uses to protect its "
        "own confidential information, but in no event less than reasonable care.\n\n"
        "2.4  Need-to-Know Basis. The Receiving Party may disclose Confidential "
        "Information only to its employees, contractors, and advisors who have a "
        "need to know such information for the Permitted Purpose and who are bound by "
        "confidentiality obligations at least as protective as those in this Agreement."
    ))

    _section(pdf, "SECTION 3 - EXCLUSIONS FROM CONFIDENTIALITY", (
        "3.1  The obligations of this Agreement shall not apply to information that:\n\n"
        "(a) Is or becomes generally known to the public through no fault of the "
        "Receiving Party;\n\n"
        "(b) Was known to the Receiving Party prior to its disclosure, as evidenced by "
        "written records pre-dating the disclosure;\n\n"
        "(c) Is independently developed by the Receiving Party without use of or "
        "reference to the Confidential Information;\n\n"
        "(d) Is lawfully received from a third party who is not bound by any "
        "confidentiality obligation;\n\n"
        "(e) Is required to be disclosed by applicable law, regulation, or court order, "
        "provided the Receiving Party gives Disclosing Party prompt written notice."
    ))

    _section(pdf, "SECTION 4 - TERM", (
        "4.1  This Agreement shall remain in effect for a period of three (3) years "
        "from the Effective Date, unless earlier terminated by mutual written consent "
        "of the parties.\n\n"
        "4.2  The confidentiality obligations with respect to any Confidential "
        "Information disclosed during the term shall survive for five (5) years "
        "after the date of such disclosure."
    ))

    _section(pdf, "SECTION 5 - RETURN OR DESTRUCTION OF MATERIALS", (
        "Upon written request by the Disclosing Party or upon termination of this "
        "Agreement, the Receiving Party shall promptly return or destroy (at the "
        "Disclosing Party's election) all tangible materials containing Confidential "
        "Information and certify in writing that all electronic copies have been "
        "permanently deleted."
    ))

    _section(pdf, "SECTION 6 - REMEDIES", (
        "6.1  The parties acknowledge that any breach of this Agreement may cause "
        "irreparable harm for which monetary damages would be an inadequate remedy. "
        "Accordingly, the non-breaching party shall be entitled to seek equitable "
        "relief, including injunction and specific performance, in addition to all "
        "other remedies available at law or in equity."
    ))

    _section(pdf, "SECTION 7 - GOVERNING LAW AND JURISDICTION", (
        "This Agreement is governed by the laws of the State of California, without "
        "regard to its conflict of laws provisions. The parties irrevocably submit to "
        "the exclusive jurisdiction of the state and federal courts located in "
        "San Francisco County, California."
    ))

    _section(pdf, "SIGNATURES", (
        "TechCorp Inc\n"
        "By: Sarah Chen, Chief Legal Officer\n"
        "Date: March 15, 2024\n\n"
        "DataVault LLC\n"
        "By: Marcus Williams, Managing Member\n"
        "Date: March 15, 2024"
    ))

    pdf.output(str(out))
    print(f"  Created: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Document 3 - Service Level Agreement
# ---------------------------------------------------------------------------

def create_sla() -> Path:
    out = RAW_DIR / "service_level_agreement.pdf"
    pdf = _pdf("SERVICE LEVEL AGREEMENT")

    _section(pdf, "PARTIES", (
        "This Service Level Agreement ('SLA' or 'Agreement') is entered into as of "
        "June 1, 2024 ('Effective Date'), by and between CloudHost Services, Inc., "
        "a Texas corporation ('Provider'), and Retail Solutions Inc., a Florida "
        "corporation ('Customer')."
    ))

    _section(pdf, "SECTION 1 - SERVICES", (
        "1.1  Scope of Services. Provider shall deliver the following managed cloud "
        "hosting services to Customer ('Services'):\n\n"
        "(a) Dedicated cloud compute instances (8 vCPU, 32 GB RAM) hosted in "
        "Provider's US-East data center;\n\n"
        "(b) Managed PostgreSQL database cluster with automated daily backups retained "
        "for ninety (90) days;\n\n"
        "(c) DDoS protection and Web Application Firewall (WAF) monitoring;\n\n"
        "(d) 24/7 infrastructure monitoring with automated alerting;\n\n"
        "(e) Up to forty (40) hours per quarter of technical support ('Support Hours')."
    ))

    _section(pdf, "SECTION 2 - SERVICE LEVELS AND UPTIME GUARANTEE", (
        "2.1  Uptime Commitment. Provider guarantees that the Services will be available "
        "ninety-nine point nine percent (99.9%) of each calendar month, calculated "
        "as: (Total Minutes - Downtime Minutes) / Total Minutes * 100.\n\n"
        "2.2  Downtime Definition. 'Downtime' means any period during which the "
        "Customer's production environment is entirely inaccessible due to Provider's "
        "infrastructure failure. Downtime excludes: scheduled maintenance windows, "
        "Customer-caused outages, and force majeure events.\n\n"
        "2.3  Scheduled Maintenance. Provider may perform scheduled maintenance between "
        "02:00 and 04:00 UTC on the first Sunday of each month. Provider shall provide "
        "at least seventy-two (72) hours advance written notice of maintenance windows.\n\n"
        "2.4  Response Times. Provider shall respond to support tickets based on "
        "severity:\n\n"
        "  - Critical (P1, complete outage): Initial response within fifteen (15) "
        "minutes; resolution target four (4) hours.\n\n"
        "  - High (P2, major feature unavailable): Initial response within one (1) "
        "hour; resolution target eight (8) hours.\n\n"
        "  - Medium (P3, partial degradation): Initial response within four (4) "
        "hours; resolution target twenty-four (24) hours.\n\n"
        "  - Low (P4, minor issue): Initial response within one (1) business day; "
        "resolution target five (5) business days."
    ))

    _section(pdf, "SECTION 3 - SLA CREDITS AND PENALTIES", (
        "3.1  SLA Credits. If Provider fails to meet the 99.9% uptime commitment in "
        "any calendar month, Customer shall be entitled to service credits as follows:\n\n"
        "  - 99.0% - 99.9% uptime: Credit equal to ten percent (10%) of monthly fee.\n\n"
        "  - 95.0% - 99.0% uptime: Credit equal to twenty-five percent (25%) of "
        "monthly fee.\n\n"
        "  - Below 95.0% uptime: Credit equal to fifty percent (50%) of monthly fee.\n\n"
        "3.2  Maximum Credit. SLA credits shall not exceed fifty percent (50%) of the "
        "monthly fee in any given month and shall constitute Customer's sole and "
        "exclusive remedy for Provider's failure to meet the uptime commitment.\n\n"
        "3.3  Credit Request. Customer must submit a credit request within thirty (30) "
        "days of the end of the affected month. Credits will be applied to the next "
        "invoice."
    ))

    _section(pdf, "SECTION 4 - FEES AND PAYMENT", (
        "4.1  Monthly Fee. Customer shall pay Provider a monthly fee of Five Thousand "
        "Dollars ($5,000.00), invoiced on the first (1st) day of each calendar month "
        "and due within thirty (30) days of invoice date.\n\n"
        "4.2  Late Payment. Amounts not paid by the due date shall accrue interest at "
        "one and one-half percent (1.5%) per month or the maximum rate permitted by "
        "applicable law, whichever is less.\n\n"
        "4.3  Price Adjustments. Provider may increase fees upon ninety (90) days "
        "written notice, but increases shall not exceed five percent (5%) per year "
        "during the initial term."
    ))

    _section(pdf, "SECTION 5 - TERM AND RENEWAL", (
        "5.1  Initial Term. This Agreement shall have an initial term of two (2) years "
        "commencing on the Effective Date ('Initial Term').\n\n"
        "5.2  Automatic Renewal. Upon expiration of the Initial Term, this Agreement "
        "shall automatically renew for successive one-year periods unless either party "
        "provides written notice of non-renewal at least sixty (60) days before the "
        "end of the then-current term.\n\n"
        "5.3  Termination for Cause. Either party may terminate this Agreement "
        "immediately upon written notice if the other party materially breaches this "
        "Agreement and fails to cure such breach within thirty (30) days of written "
        "notice of the breach."
    ))

    _section(pdf, "SECTION 6 - DATA SECURITY AND PRIVACY", (
        "6.1  Security Standards. Provider shall maintain industry-standard security "
        "measures including: SOC 2 Type II certification, AES-256 encryption at rest, "
        "TLS 1.3 encryption in transit, and multi-factor authentication for all "
        "administrative access.\n\n"
        "6.2  Data Breach Notification. In the event of a confirmed data breach "
        "affecting Customer data, Provider shall notify Customer within seventy-two "
        "(72) hours of discovery."
    ))

    _section(pdf, "SECTION 7 - GOVERNING LAW", (
        "This Agreement shall be governed by the laws of the State of Texas. Any "
        "disputes shall be resolved by binding arbitration in Austin, Texas under the "
        "JAMS Streamlined Arbitration Rules."
    ))

    _section(pdf, "SIGNATURES", (
        "CloudHost Services, Inc.\n"
        "By: Linda Park, VP of Sales\n"
        "Date: June 1, 2024\n\n"
        "Retail Solutions Inc.\n"
        "By: David Kumar, Chief Operating Officer\n"
        "Date: June 1, 2024"
    ))

    pdf.output(str(out))
    print(f"  Created: {out.name}")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def create_all_sample_docs() -> list[Path]:
    print(f"Creating sample legal documents in {RAW_DIR} ...")
    paths = [
        create_employment_contract(),
        create_nda(),
        create_sla(),
    ]
    print(f"Done - {len(paths)} documents created.")
    return paths


if __name__ == "__main__":
    create_all_sample_docs()
