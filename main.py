from text_preprocess import chunking_and_preprocessing, section_classifier, clean_jd, jd_embedding
from comparison import compare_JDandResume
import re
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
model = SentenceTransformer('all-mpnet-base-v2')


def forward_pass(resume_text,jd_text):
    def clean_jd(text):
        text = text.lower()
        text = re.sub(r"\n+", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()
    jd_text = clean_jd(jd_text)

    def jd_embedding(jd):
        jd_emb = model.encode([jd], normalize_embeddings=True)
        return jd_emb

    jd_emb=jd_embedding(jd_text)


    resume_text=chunking_and_preprocessing(resume_text)
    resume_text=section_classifier(resume_text,jd_text)
    final_score_section=compare_JDandResume(resume_text,jd_text)

    def no_section_score(resume_text):
        all_text = " ".join(
            resume_text["skills"] +
            resume_text["experience"] +
            resume_text["project"] +
            resume_text["education"]
        )

        emb = model.encode([all_text], normalize_embeddings=True)
        score = cosine_similarity(emb, jd_emb)[0][0]
        return score
    no_sec_score=no_section_score(resume_text)

    sections_to_check = ["skills", "experience", "project", "education"]

    empty_count = sum(1 for s in sections_to_check if len(resume_text[s]) == 0)

    use_section = empty_count < 2

    if use_section:
        final_score = 0.85 * no_sec_score + 0.15 * final_score_section

    else:
        final_score = no_sec_score


    final_score = min(final_score, 1)
    print(final_score_section, "with sec")
    print(no_sec_score,"no sec")
    return final_score

