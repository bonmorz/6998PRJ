# Note: you need to be using OpenAI Python v0.27.0 for the code below to work
import openai
import os
import json
from termcolor import cprint
from tqdm import tqdm
import time

openai.api_key_path = "key.env"

few_shot_examples = """
The following are multiple-choice questions about medical knowledge. Generate detailed step-by-step explanations for each question and answer.

Question: A 22-year-old male marathon runner presents to the office with the complaint of right sided rib pain when he runs long distances. Physical examination reveals normal heart and lung findings and an exhalation dysfunction at ribs 4-5 on the right. Which of the following muscles or muscle groups will be most useful in correcting this dysfunction utilizing a direct method?
(A) anterior scalene (B) latissimus dorsi (C) pectoralis minor (D) quadratus lumborum
Answer: (C)
Explanation: We refer to Wikipedia articles on medicine for help. Among the options, only pectoralis minor muscle origins from the outer surfaces of the 3rd to 5th ribs.

Question: A 36-year-old male presents to the office with a 3-week history of low back pain. He denies any recent trauma but says that he climbs in and out of his truck numerous times a day for his job. Examination of the patient in the prone position reveals a deep sacral sulcus on the left, a posterior inferior lateral angle on the right, and a lumbosacral junction that springs freely on compression. The most likely diagnosis is
(A) left-on-left sacral torsion (B) left-on-right sacral torsion (C) right unilateral sacral flexion (D) right-on-right sacral torsion
Answer: (D)
Explanation: We refer to Wikipedia articles on medicine for help. The deep sulcus on the left, a posterior ILA on the right, with a negative spring test suggests a right-on-right sacral torsion. All other options have a deep sulcus on the right.

Question: A 44-year-old man comes to the office because of a 3-day history of sore throat, nonproductive cough, runny nose, and frontal headache. He says the headache is worse in the morning and ibuprofen does provide some relief. He has not had shortness of breath. Medical history is unremarkable. He takes no medications other than the ibuprofen for pain. Vital signs are temperature 37.4°C (99.4°F), pulse 88/min, respirations 18/min, and blood pressure 120/84 mmHg. Examination of the nares shows erythematous mucous membranes. Examination of the throat shows erythema and follicular lymphoid hyperplasia on the posterior oropharynx. There is no palpable cervical adenopathy. Lungs are clear to auscultation. Which of the following is the most likely cause of this patient’s symptoms?
(A) Allergic rhinitis (B) Epstein-Barr virus (C) Mycoplasma pneumonia (D) Rhinovirus
Answer: (D)
Explanation: We refer to Wikipedia articles on medicine for help. The symptoms, especially the headache, suggest that the most likely cause is Rhinovirus. Epstein-Barr virus will cause swollen lymph nodes but there is no palpable cervical adenopathy. Lungs are clear to auscultation suggests it’s not Mycoplasma pneumonia.

Question: A previously healthy 32-year-old woman comes to the physician 8 months after her husband was killed in a car crash. Since that time, she has had a decreased appetite and difficulty falling asleep. She states that she is often sad and cries frequently. She has been rechecking the door lock five times before leaving her house and has to count exactly five pieces of toilet paper before she uses it. She says that she has always been a perfectionist but these urges and rituals are new. Pharmacotherapy should be targeted to which of the following neurotransmitters?
(A) Dopamine (B) Glutamate (C) Norepinephrine (D) Serotonin
Answer: (D)
Explanation: We refer to Wikipedia articles on medicine for help. The patient feels sad and among the options, only Dopamine and Serotonin can help increase positive emotions. Serotonin also affects digestion and metabolism, which can help the patient’s decreased appetite and sleep difficulty.

Question: A 42-year-old man comes to the office for preoperative evaluation prior to undergoing adrenalectomy scheduled in 2 weeks. One month ago, he received care in the emergency department for pain over his right flank following a motor vehicle collision. At that time, blood pressure was 160/100 mm Hg and CT scan of the abdomen showed an incidental 10-cm left adrenal mass. Results of laboratory studies, including complete blood count, serum electrolyte concentrations, and liver function tests, were within the reference ranges. The patient otherwise had been healthy and had never been told that he had elevated blood pressure. He takes no medications. A follow-up visit in the office 2 weeks ago disclosed elevated urinary normetanephrine and metanephrine and plasma aldosterone concentrations. The patient was referred to a surgeon, who recommended the adrenalectomy. Today, vital signs are temperature 36.6°C (97.9°F), pulse 100/min, respirations 14/min, and blood pressure 170/95 mm Hg. Physical examination discloses no significant findings. Initial preoperative preparation should include treatment with which of the following?
(A) Labetalol (B) A loading dose of potassium chloride (C) Nifedipine (D) Phenoxybenzamine
Answer: (D)
Explanation: We refer to Wikipedia articles on medicine for help. The symptoms and the adrenal mass suggested pheochromocytoma, and the blood pressure indicates hypertension. Phenoxybenzamine is used to treat hypertension caused by pheochromocytoma.

"""

completion_template = "Question: {}\n(A) {} (B) {} (C) {} (D) {}\nAnswer: ({})\nExplanation: "

def call_model(prompt):
    while True:
        try:
            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "user", "content": prompt},
                ],
                n=10,
            )
            cprint("PROMPT", 'blue')
            print(prompt)
            cprint("Call ChatGPT complete...", 'red')
            print(response)
            break
        except:
            print("Pausing... due to OpenAI api lag")
            time.sleep(1)
            continue
    return response

dataset_path = "./data/medqa_usmle_hf"
filenames = ["train.json"]
options = ["A", "B", "C", "D"]

save_folder = "output/medqa_usmle_hf"

for filename in filenames:
    save_path = os.path.join(save_folder, filename.split(".")[0])

    os.makedirs(save_path, exist_ok=True)

    current_id = 0
    # Implement Loading from folder
    with open(os.path.join(dataset_path, filename)) as f:
        dataset = [json.loads(data) for data in f.readlines()]

    for data in tqdm(dataset):
        out_filename = str(current_id).zfill(5) + ".json"
        if os.path.exists(os.path.join(save_path, out_filename)):
            cprint(f"{os.path.join(save_path, out_filename)} ... Already Exist", 'red')
            current_id += 1
            continue

        q = completion_template.format(data["sent1"], data["ending0"], data["ending1"], data["ending2"], data["ending3"], options[int(data["label"])])

        prompt = few_shot_examples + q

        response = call_model(prompt)
        with open(os.path.join(save_path, out_filename), 'w') as f:
            json.dump(response, f, indent=4)
        current_id += 1
