import json
import unittest

from study_planner.evaluation import (
    StudentProfile,
    SubjectExam,
    create_run_record,
    evaluate_plan,
)


class EvaluationTests(unittest.TestCase):
    def test_evaluate_plan_uses_profile_hour_limit_instead_of_hardcoded_four(self):
        profile = StudentProfile(
            subjects=[
                SubjectExam(name="NLP", exam_day=14, difficulty="hard"),
                SubjectExam(name="Deep Learning", exam_day=12, difficulty="hard"),
            ],
            daily_available_hours=5,
        )
        result_text = "\n".join(
            [
                "Day 1: NLP - 3 hours; Deep Learning - 2 hours",
                "Day 2: NLP - 6 hours",
                "Quality Score: 8/10",
            ]
        )

        evaluation = evaluate_plan(result_text, profile)

        self.assertEqual(evaluation.parsed_days[0].total_hours, 5.0)
        self.assertEqual(evaluation.parsed_days[1].total_hours, 6.0)
        self.assertEqual(evaluation.scores["constraint_satisfaction"], 0.5)
        self.assertEqual(evaluation.scores["quality"], 0.8)

    def test_create_run_record_serializes_structured_json(self):
        profile = StudentProfile(
            subjects=[
                SubjectExam(name="NLP", exam_day=14, difficulty="hard"),
                SubjectExam(name="Deep Learning", exam_day=12, difficulty="hard"),
                SubjectExam(name="Security", exam_day=10, difficulty="medium"),
                SubjectExam(name="Database", exam_day=8, difficulty="medium"),
            ],
            daily_available_hours=4,
        )
        result_text = "\n".join(
            [
                "Day 1: NLP - 2 hours; Deep Learning - 2 hours",
                "Day 2: Security - 2 hours; Database - 2 hours",
                "Quality Score: 9/10",
            ]
        )

        record = create_run_record(
            variation_name="temperature_0_2",
            temperature=0.2,
            model="gpt-4o-mini",
            raw_result=result_text,
            profile=profile,
        )
        payload = json.loads(json.dumps(record.to_json_dict()))

        self.assertEqual(payload["variation_name"], "temperature_0_2")
        self.assertEqual(payload["temperature"], 0.2)
        self.assertEqual(payload["profile"]["subjects"][1]["name"], "Deep Learning")
        self.assertEqual(payload["evaluation"]["parsed_days"][0]["day"], 1)
        self.assertEqual(payload["evaluation"]["scores"]["quality"], 0.9)


if __name__ == "__main__":
    unittest.main()
