import json
import pandas as pd
import re
from pandas import DataFrame
from sentence_transformers import SentenceTransformer
import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import spearmanr
import pytesseract


model = SentenceTransformer('all-mpnet-base-v2')


def compare_JDandResume(section_dict,jd):
    skills=section_dict['skills']
    experience=section_dict['experience']
    education=section_dict['education']
    project=section_dict['project']

    skills_emb = model.encode(skills, normalize_embeddings=True) if skills else []
    experience_emb=model.encode(experience,normalize_embeddings=True) if experience else []
    education_emb=model.encode(education,normalize_embeddings=True) if education else []
    project_emb=model.encode(project, normalize_embeddings=True) if project else []

    def section_score(emb, jd_emb):
        if len(emb) == 0:
            return 0

        sims = cosine_similarity(emb, jd_emb).flatten()
        #k = max(3, int(0.3 * len(sims)))

        #k = min(k, len(sims))
        #top_k = np.sort(sims)[-k:]  #take the last k elements
        weights = np.exp(sims)  # boosts high sims
        weights /= np.sum(weights)

        return np.sum(weights*sims)

    jd_skill_focus = "skills requirements technologies tools frameworks"
    jd_exp_focus = "responsibilities experience work tasks projects"

    jd_skill_emb = model.encode([jd + " " + jd_skill_focus], normalize_embeddings=True)
    jd_exp_emb = model.encode([jd + " " + jd_exp_focus], normalize_embeddings=True)

    sVj = section_score(skills_emb, jd_skill_emb)
    exVj = section_score(experience_emb, jd_exp_emb)
    prVj = section_score(project_emb, jd_exp_emb)  # projects behave like experience
    edVj = section_score(education_emb, jd_skill_emb)

    skills_weight = 0.35 + 0.25 * sVj
    exp_weight = 0.30 + 0.25 * exVj
    proj_weight = 0.20 + 0.15 * prVj
    edu_weight = 0.10 + 0.05 * edVj

    total = skills_weight + exp_weight + proj_weight + edu_weight

    skills_weight /= total
    exp_weight /= total
    proj_weight /= total
    edu_weight /= total

    weights = {
        "skills": skills_weight,
        "experience": exp_weight,
        "project": proj_weight,
        "education": edu_weight
    }


    final_score = (
            weights["skills"] * sVj +
            weights["experience"] * exVj +
            weights["project"] * prVj +
            weights["education"] * edVj
    )
    final_score = np.clip(final_score, -1, 1)
    return final_score








