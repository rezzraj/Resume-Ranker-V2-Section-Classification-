import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import copy

model = SentenceTransformer('all-mpnet-base-v2')


def chunking_and_preprocessing(text):

    #consist of 4 (0-1) steps

    # STEP - 0

    #preprocess text
    text = re.sub(r'\n+', '\n', text)
    # remove weird characters
    text = re.sub(r'[^\w\s,]', '', text)


    # make text lowercase
    text = text.lower()

    # STEP - 1
    #spit at every new line (\n), fullstops (.) , semicolon (;) , asterisk (*) , and hyphen (-)
    #returns a list with the split chunks as elements, the elements are string dtype
    chunks = re.split(r'[\n•\-\*\.;]+', text)
    #strips the final list and filters on the bases of non empty element of the list (if an element has nothing in it)
    chunks = [c.strip() for c in chunks if c.strip()]
    chunks = [re.sub(r'^[a-z]\s+', '', c) for c in chunks]


    # STEP - 2 removing unnecessary short words
    IMPORTANT_SHORT_WORDS = {"ml", "ai", "nlp", "sql", "c", "c++", "r"}
    def valid_chunks(chunk):
        chunk=chunk.strip().lower()

        if chunk in IMPORTANT_SHORT_WORDS:
            return True
        if len(chunk)<3:
            return False

        #removing everything except letters and no
        if not re.search(r'[a-z0-9]', chunk):
            return False
        return True

    chunks=[chunk for chunk in chunks if valid_chunks(chunk)]

    # STEP - 3 combining useful chunks

    #we will merge the chunks that are small and the chunks that contains project  together
    def consists_projects(chunk):
        #if chunk contains these verbs they are prolly talking about projects
        verbs = ["built", "developed", "worked", "created", "designed", "implemented", "improved","tuned","programmed","deployed","project"]
        return any(v in chunk for v in verbs)


    merged_chunks = []
    buffer = ""

    for chunk in chunks:

        if not buffer:
            buffer = chunk
            continue

        if (
                (len(chunk.split()) <= 2 and len(buffer.split()) <= 6)
                or
                (consists_projects(chunk) and consists_projects(buffer))
        ):
            buffer += " " + chunk
        else:
            merged_chunks.append(buffer)
            buffer = chunk

    if buffer:
        merged_chunks.append(buffer)

    return merged_chunks



def section_classifier(chunk_list,jd_text):
    sections = {
        "skills": [],
        "experience": [],
        "education": [],
        "project": [],
        "summary": []
    }

    base_anchors = {
        "skills": [
            "programming languages software tools technologies frameworks libraries technical stack",
            "development tools version control cloud platforms databases apis",
            "software engineering coding debugging testing deployment",
        ],
        "experience": [
            "work experience professional experience internship job responsibilities tasks role",
            "developed built implemented designed deployed maintained optimized applications systems",
            "collaborated team delivered features solved problems improved performance",
        ],
        "education": [
            "education degree university college academic background qualifications",
            "bachelor master computer science engineering coursework cgpa",
        ],
        "project": [
            "projects personal project academic project application system implementation",
            "built developed created application system project github deployment",
        ]
    }
    domain_anchors = {

        "frontend": {
            "skills": [
                "javascript html css typescript frontend web development react nextjs angular vue",
                "ui ux responsive design browser dom events state management redux context api",
                "frontend tools webpack vite babel tailwind bootstrap",
            ],
            "experience": [
                "developed frontend applications user interfaces web apps using react nextjs",
                "built reusable components optimized performance handled api integration",
            ],
            "project": [
                "frontend project web application react nextjs ui implementation responsive design",
            ]
        },

        "backend": {
            "skills": [
                "nodejs express django flask spring boot backend development apis microservices",
                "databases sql nosql mongodb postgresql redis",
                "authentication authorization jwt sessions server architecture",
            ],
            "experience": [
                "developed backend systems apis handled database operations server logic",
                "built scalable services optimized performance integrated systems",
            ],
            "project": [
                "backend project api development database system server implementation",
            ]
        },

        "ml": {
            "skills": [
                "machine learning deep learning nlp computer vision data science",
                "pytorch tensorflow scikit-learn pandas numpy model training evaluation",
                "llms transformers rag langchain huggingface",
            ],
            "experience": [
                "developed machine learning models trained evaluated deployed pipelines",
                "worked on data preprocessing feature engineering model optimization",
            ],
            "project": [
                "machine learning project model prediction classification nlp cv system",
            ]
        },

        "devops": {
            "skills": [
                "docker kubernetes aws gcp azure ci cd deployment cloud infrastructure",
                "monitoring logging scaling containers orchestration",
            ]
        }
    }

    detected_domain = None

    domain_anchor_embeddings = {}

    for domain, sec_dict in domain_anchors.items():
        all_text = []
        for sec in sec_dict:
            all_text.extend(sec_dict[sec])

        domain_anchor_embeddings[domain] = model.encode(
            [" ".join(all_text)], normalize_embeddings=True
        )

    def detect_domain_from_jd(jd_text, domain_anchor_embeddings):
        jd_emb = model.encode([jd_text], normalize_embeddings=True)

        domain_scores = {}

        for domain, domain_emb in domain_anchor_embeddings.items():
            score = cosine_similarity(domain_emb, jd_emb)[0][0]
            domain_scores[domain] = score

        best_domain = max(domain_scores, key=domain_scores.get)

        return best_domain, domain_scores


    if jd_text:
        detected_domain, domain_scores = detect_domain_from_jd(jd_text, domain_anchor_embeddings)

        print("DOMAIN SCORES:", domain_scores)
        print("DETECTED DOMAIN:", detected_domain)



    final_anchors = copy.deepcopy(base_anchors)

    if detected_domain and detected_domain in domain_anchors:
        for sec in domain_anchors[detected_domain]:
            final_anchors[sec] += domain_anchors[detected_domain][sec]







    def make_anchor_emb(anchor_list):
        embs = model.encode(anchor_list,normalize_embeddings=True)
        return np.mean(embs, axis=0)

    #turning anchors into embedding
    skills_emb = make_anchor_emb(final_anchors["skills"])
    experience_emb = make_anchor_emb(final_anchors["experience"])
    education_emb = make_anchor_emb(final_anchors["education"])
    project_emb = make_anchor_emb(final_anchors["project"])



    #now using the anchors (final anchors)

    chunk_embs = model.encode(chunk_list,normalize_embeddings=True)

    current_section = None
    for chunk, chunk_emb in zip(chunk_list, chunk_embs):

        #  ADD HERE
        # skip junk, header, contact
        if any(x in chunk for x in ["+91", "linkedin", "github", "email", "address"]):
            continue

        # skip soft skills
        if any(x in chunk for x in ["communication", "teamwork", "adaptability", "leadership"]):
            continue

        # we first check using hard code if chunk contains any of these words its likely talking about stuff that fits in those category
        if "skills" in chunk:
            current_section = "skills"
            continue
        elif "education" in chunk:
            current_section = "education"
            continue
        elif "project" in chunk:
            current_section = "project"
            continue
        elif "experience" in chunk:
            current_section = "experience"
            continue
        if "summary" in chunk:
            continue

        if any(word in chunk for word in ["engineer", "intern", "worked at", "company"]):
            sections["experience"].append(chunk)
            continue

        if current_section:
            sections[current_section].append(chunk)
            continue







        #compare
        skills_cs = cosine_similarity([chunk_emb], [skills_emb])[0][0]
        exp_cs = cosine_similarity([chunk_emb], [experience_emb])[0][0]
        edu_cs = cosine_similarity([chunk_emb], [education_emb])[0][0]
        project_cs = cosine_similarity([chunk_emb], [project_emb])[0][0]

        scores = {
            "skills": skills_cs,
            "experience": exp_cs,
            "education": edu_cs,
            "project": project_cs
        }

        max_key = max(scores,key=scores.get)
        max_value = scores[max_key]


        sorted_scores=sorted(scores.values(),reverse=True)
        margin= sorted_scores[0]-sorted_scores[1]

        if max_value > 0.5 and margin > 0.08:
            sections[max_key].append(chunk)
        else:
            continue




    return sections



def jd_embedding(jd):
    jd_emb = model.encode([jd],normalize_embeddings=True)
    return jd_emb




def clean_jd(text):
    text = text.lower()
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()







def no_section_score(data,jd):
    all_text = " ".join(
        data["skills"] +
        data["experience"] +
        data["project"] +
        data["education"]
    )

    emb = model.encode([all_text], normalize_embeddings=True)
    jd_emb = model.encode([jd], normalize_embeddings=True)
    score = cosine_similarity(emb, jd_emb)[0][0]
    return score













