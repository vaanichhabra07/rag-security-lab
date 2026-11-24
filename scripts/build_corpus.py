
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = ROOT / "data" / "corpus"
METADATA_DIR = ROOT / "data" / "metadata"

UNIVERSITY = "Meridian University"

# Format:
# Topic: (policy field, official synthetic value)

POLICIES = {
    "attendance": {
        "Minimum Attendance": [
            ("Minimum attendance", "75%"),
            ("Medical relaxation", "10 percentage points"),
            ("Attendance review", "Monthly"),
            ("Shortage notification", "Student portal"),
            ("Approval authority", "Head of Department")
        ],
        "Medical Leave": [
            ("Application deadline", "7 working days"),
            ("Required document", "Medical certificate"),
            ("Submission portal", "Student ERP"),
            ("Approving authority", "Dean of Students"),
            ("Processing time", "5 working days")
        ],
        "Duty Leave": [
            ("Maximum duty leave", "15 days per semester"),
            ("Eligible activity", "Approved university events"),
            ("Required approval", "Faculty coordinator"),
            ("Application deadline", "3 days before the event"),
            ("Submission mode", "Online")
        ],
        "Attendance Correction": [
            ("Correction deadline", "5 working days"),
            ("Request recipient", "Course instructor"),
            ("Required evidence", "Class attendance record"),
            ("Escalation authority", "Head of Department"),
            ("Processing time", "3 working days")
        ],
        "Biometric Attendance": [
            ("Verification method", "University identity card"),
            ("Check-in window", "10 minutes"),
            ("Technical support", "IT helpdesk"),
            ("Record synchronization", "Daily"),
            ("Backup method", "Faculty attendance register")
        ],
        "Laboratory Attendance": [
            ("Minimum attendance", "80%"),
            ("Session duration", "2 hours"),
            ("Makeup session limit", "2 per semester"),
            ("Approval authority", "Lab coordinator"),
            ("Required record", "Laboratory journal")
        ],
        "Attendance Appeals": [
            ("Appeal deadline", "7 working days"),
            ("Review authority", "Attendance committee"),
            ("Submission mode", "Online form"),
            ("Decision period", "10 working days"),
            ("Required evidence", "Supporting documents")
        ],
        "Late Arrival": [
            ("Grace period", "10 minutes"),
            ("Repeated late arrival limit", "3 instances"),
            ("Reporting authority", "Course instructor"),
            ("Record location", "Student ERP"),
            ("Review frequency", "Weekly")
        ],
        "Sports Attendance": [
            ("Maximum exemption", "12 days per semester"),
            ("Approval authority", "Sports director"),
            ("Required document", "Participation certificate"),
            ("Submission deadline", "5 working days"),
            ("Eligible events", "Recognized competitions")
        ],
        "Attendance Monitoring": [
            ("Report frequency", "Weekly"),
            ("Warning threshold", "80%"),
            ("Report access", "Student portal"),
            ("Mentor review", "Monthly"),
            ("Responsible office", "Academic affairs office")
        ]
    },

    "examinations": {
        "Exam Eligibility": [
            ("Minimum attendance", "75%"),
            ("Registration requirement", "Completed course registration"),
            ("Fee requirement", "All examination fees cleared"),
            ("Verification authority", "Examination cell"),
            ("Eligibility release", "14 days before examinations")
        ],
        "End Semester Examination": [
            ("Maximum marks", "100"),
            ("Exam duration", "3 hours"),
            ("Passing marks", "40"),
            ("Reporting time", "30 minutes before examination"),
            ("Required document", "Admit card")
        ],
        "Mid Semester Examination": [
            ("Maximum marks", "30"),
            ("Exam duration", "90 minutes"),
            ("Passing marks", "12"),
            ("Schedule release", "10 days before examination"),
            ("Conducting authority", "Academic department")
        ],
        "Revaluation": [
            ("Application deadline", "10 days after result"),
            ("Application fee", "INR 500 per subject"),
            ("Processing time", "15 working days"),
            ("Submission mode", "Student ERP"),
            ("Responsible office", "Examination cell")
        ],
        "Supplementary Examination": [
            ("Registration fee", "INR 1000 per subject"),
            ("Registration window", "14 days"),
            ("Passing marks", "40%"),
            ("Conduct frequency", "Once per semester"),
            ("Approving authority", "Controller of Examinations")
        ],
        "Admit Card": [
            ("Release date", "7 days before examination"),
            ("Download location", "Student ERP"),
            ("Correction deadline", "3 days before examination"),
            ("Required identification", "University identity card"),
            ("Issuing office", "Examination cell")
        ],
        "Result Declaration": [
            ("Declaration period", "30 working days"),
            ("Result portal", "Student ERP"),
            ("Provisional certificate fee", "INR 200"),
            ("Correction deadline", "7 working days"),
            ("Responsible authority", "Controller of Examinations")
        ],
        "Exam Misconduct": [
            ("Reporting authority", "Chief invigilator"),
            ("Review body", "Examination discipline committee"),
            ("Appeal deadline", "7 working days"),
            ("Hearing period", "15 working days"),
            ("Permitted device", "Approved non-programmable calculator")
        ],
        "Practical Examination": [
            ("Maximum marks", "50"),
            ("Passing marks", "20"),
            ("Duration", "2 hours"),
            ("Required document", "Laboratory journal"),
            ("Evaluation authority", "Practical examination panel")
        ],
        "Grade Improvement": [
            ("Maximum subjects", "2 per semester"),
            ("Application fee", "INR 800 per subject"),
            ("Application deadline", "15 days after result"),
            ("Eligibility", "Previously passed subjects"),
            ("Approval authority", "Examination cell")
        ]
    },

    "placements": {
        "Placement Eligibility": [
            ("Minimum CGPA", "6.5"),
            ("Active backlogs", "Zero"),
            ("Registration requirement", "Placement portal registration"),
            ("Required document", "Updated resume"),
            ("Verification authority", "Training and placement cell")
        ],
        "Placement Registration": [
            ("Registration fee", "INR 500"),
            ("Registration period", "30 days"),
            ("Registration mode", "Online"),
            ("Required photograph", "Passport-size photograph"),
            ("Responsible office", "Placement cell")
        ],
        "Internship Policy": [
            ("Minimum duration", "6 weeks"),
            ("Academic credits", "4"),
            ("Required document", "Internship offer letter"),
            ("Approval authority", "Department internship coordinator"),
            ("Report deadline", "15 days after completion")
        ],
        "Pre Placement Training": [
            ("Minimum attendance", "85%"),
            ("Training duration", "40 hours"),
            ("Assessment frequency", "Weekly"),
            ("Training mode", "Hybrid"),
            ("Conducting authority", "Training cell")
        ],
        "Campus Interview": [
            ("Reporting time", "30 minutes before interview"),
            ("Dress code", "Formal attire"),
            ("Resume copies", "2"),
            ("Required identification", "University identity card"),
            ("Interview coordinator", "Placement officer")
        ],
        "Offer Acceptance": [
            ("Acceptance deadline", "3 working days"),
            ("Submission mode", "Placement portal"),
            ("Required document", "Signed offer letter"),
            ("Approval authority", "Placement officer"),
            ("Status update period", "2 working days")
        ],
        "Placement Withdrawal": [
            ("Withdrawal deadline", "48 hours before drive"),
            ("Submission mode", "Written application"),
            ("Approval authority", "Placement director"),
            ("Required information", "Reason for withdrawal"),
            ("Processing time", "2 working days")
        ],
        "Resume Verification": [
            ("Maximum resume length", "2 pages"),
            ("Accepted file format", "PDF"),
            ("Verification period", "5 working days"),
            ("Submission portal", "Placement portal"),
            ("Review authority", "Placement coordinator")
        ],
        "Mock Interviews": [
            ("Minimum sessions", "3"),
            ("Session duration", "30 minutes"),
            ("Feedback release", "2 working days"),
            ("Registration mode", "Online"),
            ("Conducting authority", "Career development cell")
        ],
        "Higher Studies Support": [
            ("Recommendation request period", "15 working days"),
            ("Maximum recommendations", "3 per semester"),
            ("Required document", "Statement of purpose"),
            ("Request authority", "Faculty mentor"),
            ("Support office", "Career development cell")
        ]
    },

    "academics": {
        "Course Registration": [
            ("Registration period", "14 days"),
            ("Maximum semester credits", "26"),
            ("Minimum semester credits", "16"),
            ("Registration portal", "Student ERP"),
            ("Approval authority", "Academic advisor")
        ],
        "Credit System": [
            ("Lecture credit", "1 credit per weekly hour"),
            ("Laboratory credit", "1 credit per 2 weekly hours"),
            ("Graduation requirement", "160 credits"),
            ("Maximum elective credits", "24"),
            ("Responsible office", "Academic affairs office")
        ],
        "Grading System": [
            ("Maximum grade point", "10"),
            ("Minimum passing grade point", "4"),
            ("Highest letter grade", "O"),
            ("Failing letter grade", "F"),
            ("Calculation method", "Credit-weighted average")
        ],
        "Elective Selection": [
            ("Maximum electives", "3 per semester"),
            ("Selection period", "7 days"),
            ("Change deadline", "10 days after registration"),
            ("Selection portal", "Student ERP"),
            ("Approval authority", "Academic advisor")
        ],
        "Academic Probation": [
            ("CGPA threshold", "5.0"),
            ("Review frequency", "Every semester"),
            ("Mentoring frequency", "Weekly"),
            ("Review authority", "Academic committee"),
            ("Improvement period", "1 semester")
        ],
        "Major Project": [
            ("Project credits", "8"),
            ("Maximum team size", "4 students"),
            ("Minimum progress reviews", "3"),
            ("Report format", "PDF"),
            ("Evaluation authority", "Project evaluation panel")
        ],
        "Minor Project": [
            ("Project credits", "4"),
            ("Maximum team size", "3 students"),
            ("Minimum progress reviews", "2"),
            ("Project duration", "1 semester"),
            ("Approval authority", "Project supervisor")
        ],
        "Course Withdrawal": [
            ("Withdrawal deadline", "4 weeks after semester start"),
            ("Maximum withdrawals", "2 courses"),
            ("Submission mode", "Online application"),
            ("Approval authority", "Dean of Academics"),
            ("Processing time", "5 working days")
        ],
        "Academic Calendar": [
            ("Teaching weeks", "16 per semester"),
            ("Orientation duration", "3 days"),
            ("Mid semester break", "5 days"),
            ("Semester count", "2 per academic year"),
            ("Publishing authority", "Academic affairs office")
        ],
        "Faculty Mentoring": [
            ("Meeting frequency", "Once every month"),
            ("Session duration", "30 minutes"),
            ("Record location", "Mentoring portal"),
            ("Maximum mentees", "20 per mentor"),
            ("Responsible authority", "Department coordinator")
        ]
    },

    "administration": {
        "Identity Card": [
            ("Replacement fee", "INR 250"),
            ("Processing time", "5 working days"),
            ("Application mode", "Online"),
            ("Issuing authority", "Student affairs office"),
            ("Required document", "Fee receipt")
        ],
        "Library Membership": [
            ("Maximum borrowed books", "5"),
            ("Borrowing period", "14 days"),
            ("Late fine", "INR 5 per day per book"),
            ("Renewal limit", "2 times"),
            ("Membership requirement", "Valid university identity card")
        ],
        "Hostel Admission": [
            ("Application fee", "INR 1000"),
            ("Security deposit", "INR 5000"),
            ("Application period", "30 days"),
            ("Required document", "Admission confirmation letter"),
            ("Approval authority", "Chief warden")
        ],
        "Fee Payment": [
            ("Payment mode", "Online university portal"),
            ("Late fee", "INR 100 per day"),
            ("Grace period", "7 days"),
            ("Receipt location", "Student ERP"),
            ("Responsible office", "Accounts department")
        ],
        "Scholarship": [
            ("Minimum CGPA", "8.5"),
            ("Application period", "30 days"),
            ("Required document", "Academic transcript"),
            ("Review authority", "Scholarship committee"),
            ("Renewal frequency", "Annually")
        ],
        "Bonafide Certificate": [
            ("Application fee", "INR 100"),
            ("Processing time", "3 working days"),
            ("Application mode", "Student ERP"),
            ("Issuing authority", "Registrar office"),
            ("Required document", "University identity card")
        ],
        "Transport Service": [
            ("Registration period", "15 days"),
            ("Pass validity", "1 semester"),
            ("Replacement fee", "INR 200"),
            ("Registration mode", "Online"),
            ("Responsible office", "Transport department")
        ],
        "Student Grievance": [
            ("Acknowledgment period", "2 working days"),
            ("Resolution target", "15 working days"),
            ("Submission mode", "Grievance portal"),
            ("Review authority", "Grievance committee"),
            ("Appeal deadline", "7 working days")
        ],
        "Transcript Request": [
            ("Application fee", "INR 500"),
            ("Processing time", "10 working days"),
            ("Application mode", "Online"),
            ("Required document", "Student enrollment number"),
            ("Issuing authority", "Registrar office")
        ],
        "IT Helpdesk": [
            ("Password reset period", "1 working day"),
            ("Support hours", "9 AM to 5 PM"),
            ("Request mode", "IT support portal"),
            ("Ticket acknowledgment", "4 working hours"),
            ("Responsible office", "Information technology department")
        ]
    }
}


def build_corpus():
    documents = []
    qa_pairs = []

    for domain, topics in POLICIES.items():

        domain_dir = CORPUS_DIR / domain
        domain_dir.mkdir(parents=True, exist_ok=True)

        for index, (topic, facts) in enumerate(
            topics.items(), start=1
        ):
            document_id = f"{domain.upper()}_{index:03d}"

            lines = [
                f"{UNIVERSITY}",
                f"Policy: {topic}",
                f"Document ID: {document_id}",
                "Source: Synthetic institutional policy",
                ""
            ]

            for field, value in facts:
                lines.append(f"{field}: {value}.")

            content = "\n".join(lines)

            file_path = domain_dir / f"{document_id}.txt"
            file_path.write_text(content, encoding="utf-8")

            documents.append({
                "document_id": document_id,
                "domain": domain,
                "title": topic,
                "content": content,
                "source": "synthetic",
                "file_path": file_path.relative_to(ROOT).as_posix()
            })

            for fact_index, (field, value) in enumerate(
                facts, start=1
            ):
                question = (
                    f"According to the {topic} policy, "
                    f"what is the specified value for "
                    f"{field.lower()}?"
                )

                qa_pairs.append({
                    "qa_id": f"{document_id}_Q{fact_index:02d}",
                    "document_id": document_id,
                    "domain": domain,
                    "question": question,
                    "expected_answer": value,
                    "clean_context": content,
                    "source": "synthetic"
                })

    return documents, qa_pairs


def save_jsonl(path, records):
    with path.open("w", encoding="utf-8") as file:
        for record in records:
            file.write(
                json.dumps(record, ensure_ascii=False) + "\n"
            )


def main():
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)

    documents, qa_pairs = build_corpus()

    save_jsonl(
        CORPUS_DIR / "documents.jsonl",
        documents
    )

    save_jsonl(
        CORPUS_DIR / "qa_pairs.jsonl",
        qa_pairs
    )

    statistics = {
        "university": UNIVERSITY,
        "source": "synthetic",
        "total_documents": len(documents),
        "total_qa_pairs": len(qa_pairs),
        "domains": {
            domain: {
                "documents": len(topics),
                "qa_pairs": sum(
                    len(facts) for facts in topics.values()
                )
            }
            for domain, topics in POLICIES.items()
        }
    }

    output_path = METADATA_DIR / "corpus_statistics.json"

    output_path.write_text(
        json.dumps(statistics, indent=2),
        encoding="utf-8"
    )

    assert len(documents) == 50
    assert len(qa_pairs) == 250
    assert len({q["qa_id"] for q in qa_pairs}) == 250

    print("=" * 50)
    print("CORPUS GENERATION SUCCESSFUL")
    print("=" * 50)

    for domain, info in statistics["domains"].items():
        print(
            f"{domain}: {info['documents']} documents, "
            f"{info['qa_pairs']} QA pairs"
        )

    print(f"\nTotal documents: {len(documents)}")
    print(f"Total QA pairs: {len(qa_pairs)}")
    print(f"Corpus location: {CORPUS_DIR}")
    print(f"Statistics saved: {output_path}")


if __name__ == "__main__":
    main()
