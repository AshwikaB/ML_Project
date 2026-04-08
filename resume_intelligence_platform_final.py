# ============================================================
# RESUME INTELLIGENCE PLATFORM
# Tech Stack: Python | Gradio | scikit-learn | TF-IDF
# Domain: Tech / Software / Data Science
# Author: Built for ML Viva Project
# ============================================================

# ─────────────────────────────────────────────
# SECTION 0: INSTALLATION (Run in Colab)
# ─────────────────────────────────────────────
# !pip install gradio scikit-learn nltk PyPDF2 python-docx matplotlib seaborn datasets -q

# ─────────────────────────────────────────────
# SECTION 1: IMPORTS
# ─────────────────────────────────────────────
import os
import re
import json
import random
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import gradio as gr

# File parsing
import PyPDF2
import docx

# NLP & ML
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    f1_score, confusion_matrix, ConfusionMatrixDisplay,
    classification_report
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import cosine_similarity

# HuggingFace datasets (for real data augmentation)
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# SECTION 2: NLTK DOWNLOADS
# ─────────────────────────────────────────────
for resource in ["stopwords", "wordnet", "omw-1.4", "punkt"]:
    try:
        nltk.download(resource, quiet=True)
    except Exception:
        pass

# ─────────────────────────────────────────────
# SECTION 3: CONSTANTS & SKILL KNOWLEDGE BASE
# ─────────────────────────────────────────────

# Comprehensive Tech / Software / Data Science skill taxonomy
SKILL_TAXONOMY = {
    # Programming Languages
    "python": ["py", "python3", "python2"],
    "javascript": ["js", "ecmascript", "es6", "es2015"],
    "typescript": ["ts"],
    "java": ["java8", "java11", "java17", "jvm"],
    "c++": ["cpp", "c plus plus"],
    "c#": ["csharp", "c sharp", "dotnet", ".net"],
    "r": ["r language", "r programming"],
    "scala": ["scala lang"],
    "go": ["golang"],
    "rust": ["rust lang"],
    "kotlin": [],
    "swift": [],
    "php": [],
    "ruby": ["ruby on rails", "rails"],
    "sql": ["structured query language", "mysql", "postgresql", "sqlite", "mssql", "t-sql"],

    # ML / AI / DS
    # FIX: Removed unsafe short aliases: "ml" hits xml/html/yml; "cv" hits "CV attached"; "rl" hits url/world
    "machine learning": ["statistical learning"],
    "deep learning": ["neural networks", "ann", "dnn"],
    "natural language processing": ["nlp", "text mining", "computational linguistics"],
    "computer vision": ["image processing", "object detection"],
    "reinforcement learning": ["reward learning"],
    "data science": ["data analytics", "analytics"],
    "data engineering": ["data pipeline", "etl", "data warehousing"],
    "feature engineering": ["feature extraction", "feature selection"],
    "model deployment": ["mlops", "model serving", "productionization"],
    "generative ai": ["genai", "llm", "large language models", "gpt", "chatgpt"],

    # ML Frameworks & Libraries
    "tensorflow": ["tf", "keras"],
    "pytorch": ["torch"],
    "scikit-learn": ["sklearn", "scikit learn"],
    "xgboost": ["xgb", "extreme gradient boosting"],
    "lightgbm": ["lgbm"],
    "hugging face": ["transformers", "huggingface"],
    "opencv": ["cv2"],
    "numpy": ["np", "numerical python"],
    "pandas": ["dataframe", "data wrangling"],
    "matplotlib": ["pyplot", "plotting"],
    "seaborn": [],
    "plotly": [],
    "scipy": [],
    "statsmodels": [],

    # Cloud Platforms
    "aws": ["amazon web services", "ec2", "s3", "lambda", "sagemaker"],
    "azure": ["microsoft azure", "azure ml"],
    "google cloud": ["gcp", "google cloud platform", "bigquery", "vertex ai"],

    # MLOps / DevOps / Tools
    "docker": ["containerization", "containers"],
    "kubernetes": ["k8s", "container orchestration"],
    "git": ["github", "gitlab", "version control", "bitbucket"],
    "ci/cd": ["continuous integration", "continuous deployment", "jenkins", "github actions"],
    "airflow": ["apache airflow", "workflow orchestration"],
    "mlflow": ["ml experiment tracking"],
    "dvc": ["data version control"],
    "fastapi": ["fast api", "rest api", "api development"],
    "flask": ["flask api"],
    "django": [],

    # Data Infra
    "spark": ["apache spark", "pyspark", "big data"],
    "kafka": ["apache kafka", "event streaming"],
    "hadoop": ["hdfs", "mapreduce"],
    "databricks": [],
    "snowflake": [],
    "dbt": ["data build tool"],
    "elasticsearch": ["elastic search"],
    "redis": [],
    "mongodb": ["nosql", "document database"],
    "postgresql": ["postgres"],
    "mysql": [],

    # Concepts
    "statistics": ["statistical analysis", "probability", "hypothesis testing"],  # FIX: duplicate key removed
    "linear algebra": ["matrix operations", "vectors", "eigenvalues"],
    "optimization": ["gradient descent", "adam", "sgd"],
    "a/b testing": ["split testing", "ab testing", "a/b test"],
    "data visualization": ["dashboarding", "tableau", "power bi", "looker"],
    "agile": ["scrum", "kanban", "sprint"],
    "communication": ["presentation", "stakeholder management", "storytelling"],
    "problem solving": ["analytical thinking", "critical thinking"],
}

# Flatten synonym → canonical skill mapping
SYNONYM_MAP = {}
for canonical, synonyms in SKILL_TAXONOMY.items():
    SYNONYM_MAP[canonical] = canonical
    for syn in synonyms:
        SYNONYM_MAP[syn.lower()] = canonical

# Resume validation keywords
RESUME_REQUIRED_KEYWORDS = [
    "education", "experience", "skills", "work", "project",
    "university", "college", "degree", "bachelor", "master",
    "engineer", "developer", "analyst", "intern", "certification"
]

# ─────────────────────────────────────────────
# SECTION 4: SYNTHETIC DATASET GENERATOR
# ─────────────────────────────────────────────

def build_synthetic_dataset() -> pd.DataFrame:
    """
    Generate a scaled synthetic dataset with ~2500 samples.
    Uses domain-aware pairing and synthetic noise/randomness for diversity:
      - matched cases  (label=1)
      - unrelated cases (label=0)
      - borderline cases (label=0 or 1 by design)
    ML logic and model training are NOT changed — only data volume.
    """

    rng = random.Random(42)  # deterministic noise seed

    # ── Noise pools: slight randomness injected to vary text ──
    _exp_adjectives = ["detail-oriented", "results-driven", "self-motivated", "collaborative",
                       "innovative", "analytical", "pragmatic", "proactive"]
    _edu_variants = ["B.Tech", "M.Tech", "B.Sc.", "M.Sc.", "B.E.", "M.E.", "B.S.", "M.S."]
    _university_variants = ["IIT Bombay", "IIT Delhi", "IIT Madras", "NIT Trichy",
                            "BITS Pilani", "DTU", "VIT", "Manipal University", "SRM University"]
    _cloud_alt = ["AWS", "GCP", "Azure", "AWS/GCP", "GCP/Azure", "AWS and Azure"]
    _cert_phrases = [
        "Certifications: AWS Certified ML Specialty.",
        "Certifications: Google Professional Data Engineer.",
        "Certified: Azure Data Scientist Associate.",
        "Completed: Coursera Deep Learning Specialization.",
        "",  # sometimes no cert mentioned
    ]
    _proj_phrases = [
        "Projects: Fraud detection system ({acc}% AUC), customer segmentation pipeline, A/B test framework.",
        "Projects: Real-time recommendation engine, NLP sentiment classifier ({acc}% F1), data lake on S3.",
        "Projects: Churn prediction model ({acc}% accuracy), ETL pipeline automation, dashboard in Tableau.",
        "Projects: Image classification with ResNet ({acc}% accuracy), MLflow experiment tracking, FastAPI serving.",
        "Projects: Time-series forecasting, feature store design, automated model retraining pipeline.",
    ]

    def _noise(base: str, exp: int) -> str:
        """Inject slight randomness into a resume/JD template string."""
        adj = rng.choice(_exp_adjectives)
        edu = rng.choice(_edu_variants)
        uni = rng.choice(_university_variants)
        cloud = rng.choice(_cloud_alt)
        cert = rng.choice(_cert_phrases)
        proj = rng.choice(_proj_phrases).format(acc=rng.randint(82, 97))
        # substitute placeholders that exist in template, ignore missing ones
        result = base.format(exp=exp)
        result = result.replace("{adj}", adj).replace("{edu}", edu)
        result = result.replace("{uni}", uni).replace("{cloud}", cloud)
        result = result.replace("{cert}", cert).replace("{proj}", proj)
        # Append a small random addendum for text variation
        addendum = rng.choice([
            f" Strong {adj} professional. {cert}",
            f" {edu} from {uni}. {proj}",
            f" Cloud platforms: {cloud}. {cert}",
            f" Known for {adj} approach. {proj}",
            "",
        ])
        return result + addendum

    # ── Resume templates per role (base text, {exp} placeholder) ──
    resume_templates = {
        "data_scientist": [
            "Experienced Data Scientist with {exp} years of experience in machine learning, deep learning, and statistical modeling. "
            "Proficient in Python, TensorFlow, PyTorch, scikit-learn, and SQL. Built predictive models using XGBoost and LightGBM. "
            "Skilled in natural language processing, computer vision, and A/B testing. "
            "Education: M.Sc. in Data Science. Projects: Customer churn prediction, sentiment analysis pipeline using BERT, real-time recommendation engine.",

            "Data Scientist with {exp} years specializing in NLP and generative AI. "
            "Worked with Hugging Face transformers, LLMs, and prompt engineering. "
            "Strong Python, pandas, numpy, matplotlib skills. "
            "Experience deploying models with FastAPI and Docker on AWS SageMaker. "
            "Degree: B.Tech Computer Science. Skills: MLflow, DVC, Airflow, Spark.",

            "Junior Data Scientist with {exp} year of experience. "
            "Familiar with scikit-learn, pandas, matplotlib, and basic SQL. "
            "Completed projects on regression, classification, and clustering. "
            "Education: B.Sc. Statistics. Skills include Python, data visualization, feature engineering.",

            "Senior Data Scientist with {exp} years in e-commerce analytics and recommendation systems. "
            "Proficient in Python, Spark, SQL, XGBoost, and A/B testing. Deep experience with Tableau and Power BI. "
            "Education: M.Sc. Applied Mathematics. Strong statistical modeling and hypothesis testing background.",

            "Data Scientist with {exp} years in healthcare analytics. "
            "Built clinical NLP models using spaCy and BERT. Expertise in survival analysis, logistic regression, and random forests. "
            "Python, R, SQL. Published research on ML in medical imaging. Education: M.D./PhD equivalent.",
        ],
        "ml_engineer": [
            "Machine Learning Engineer with {exp} years building scalable ML systems. "
            "Expert in MLOps: Docker, Kubernetes, Airflow, MLflow, and CI/CD pipelines. "
            "Deployed 10+ production models on AWS and GCP. "
            "Languages: Python, Scala, Go. Frameworks: TensorFlow, PyTorch, FastAPI. "
            "Education: M.Tech AI. Strong in system design and model optimization.",

            "ML Engineer specializing in computer vision and model deployment. "
            "{exp} years experience with OpenCV, YOLO, TensorFlow Lite, and ONNX. "
            "Built edge inference pipelines for IoT devices. "
            "Skills: Python, C++, Docker, Kubernetes, Git, Azure. "
            "Education: B.E. Electronics. Certifications: AWS ML Specialty.",

            "Senior ML Engineer with {exp} years. Expert in distributed training with PyTorch DDP and Spark. "
            "Built feature stores and real-time prediction APIs. "
            "Skills: Python, Kafka, Redis, PostgreSQL, Databricks, Snowflake. "
            "Education: M.Sc. Computer Science.",

            "ML Engineer with {exp} years in NLP and conversational AI. "
            "Built transformer-based pipelines using Hugging Face and deployed on GCP Vertex AI. "
            "Strong Python, FastAPI, Docker skills. Familiar with LangChain and vector databases. "
            "Education: B.Tech Computer Science.",

            "ML Research Engineer with {exp} years. "
            "Focus on reinforcement learning and multi-agent systems. "
            "Published work on policy optimization. PyTorch, JAX, and C++ for performance-critical code. "
            "Education: M.Sc. Computer Science (AI track).",
        ],
        "data_engineer": [
            "Data Engineer with {exp} years of experience designing ETL pipelines and data warehouses. "
            "Expert in Apache Spark, Kafka, Airflow, and dbt. "
            "Built petabyte-scale data lakes on AWS S3 with Redshift and Snowflake. "
            "Languages: Python, SQL, Scala. Skills: Docker, Kubernetes, Git, CI/CD. "
            "Education: B.Tech IT.",

            "Data Engineer with {exp} years focused on real-time streaming architectures. "
            "Used Kafka, Spark Streaming, and Flink for event-driven systems. "
            "Strong in PostgreSQL, MongoDB, Elasticsearch, and Redis. "
            "Python, SQL, and cloud platforms (GCP, Azure). "
            "Degree: M.Sc. Information Systems.",

            "Junior Data Engineer with {exp} year of experience in ETL development. "
            "Skills: Python, SQL, Airflow, dbt, and basic Spark. "
            "Worked with AWS Glue and Redshift for data pipeline development. "
            "Education: B.Sc. Computer Science.",

            "Senior Data Engineer with {exp} years building cloud-native data platforms. "
            "Expert in Databricks, Delta Lake, dbt, and Snowflake. Strong Python, SQL, and Terraform skills. "
            "Led migration of legacy warehouses to modern lakehouse architecture. Education: B.Tech IT.",

            "Data Engineer with {exp} years specializing in analytics engineering. "
            "dbt, Looker, BigQuery, and Airflow expert. Built self-serve analytics platforms. "
            "Python and SQL proficiency. Experience with data quality frameworks and Great Expectations. "
            "Education: M.Sc. Statistics.",
        ],
        "software_engineer": [
            "Software Engineer with {exp} years developing scalable backend systems. "
            "Languages: Java, Python, Go. Frameworks: Spring Boot, Django, FastAPI. "
            "Worked with microservices, REST APIs, and event-driven architectures using Kafka. "
            "Skills: Docker, Kubernetes, CI/CD, PostgreSQL, Redis, Git. "
            "Education: B.Tech Computer Science.",

            "Full-Stack Software Engineer with {exp} years. "
            "Frontend: JavaScript, TypeScript, React, Next.js. Backend: Node.js, Python, Django. "
            "Databases: PostgreSQL, MongoDB, Redis. DevOps: Docker, AWS, GitHub Actions. "
            "Education: B.E. Information Technology.",

            "Backend Engineer with {exp} years specializing in high-performance systems. "
            "C++, Rust, and Python for systems programming. "
            "Experience with distributed systems, gRPC, and message queues. "
            "Education: M.Tech Computer Science.",

            "Software Engineer with {exp} years building fintech platforms. "
            "Java Spring Boot, Kafka, PostgreSQL, and Redis. Strong in TDD and system design. "
            "Experience with PCI-DSS compliance, API gateway design. Education: B.Tech CS.",

            "Mobile & Backend Engineer with {exp} years. "
            "Kotlin for Android, Swift for iOS, Python/FastAPI for backend. "
            "Firebase, MongoDB, and AWS. Experience shipping apps to 1M+ users. "
            "Education: B.Sc. Computer Science.",
        ],
        "devops_engineer": [
            "DevOps Engineer with {exp} years in cloud infrastructure and automation. "
            "Expert in Kubernetes, Docker, Terraform, Ansible, and Jenkins. "
            "Managed AWS, GCP, and Azure infrastructure. "
            "Strong in CI/CD pipelines, monitoring (Prometheus, Grafana), and security. "
            "Education: B.Tech IT.",

            "Site Reliability Engineer with {exp} years. "
            "Skills: Kubernetes, Helm, ArgoCD, Terraform, Python scripting. "
            "Incident management, capacity planning, and chaos engineering. "
            "Education: B.Sc. Computer Science.",

            "Platform Engineer with {exp} years building internal developer platforms. "
            "Expert in Backstage, GitHub Actions, Terraform, and Kubernetes. "
            "Python and Go scripting. FinOps experience reducing cloud costs by 30%. "
            "Education: B.Tech IT.",
        ],
        "unrelated": [
            "Marketing Manager with {exp} years of experience in brand strategy, digital marketing, and campaign management. "
            "Managed social media accounts, SEO optimization, and Google Ads campaigns. "
            "Education: MBA Marketing. Skills: content creation, market research, CRM tools.",

            "HR Manager with {exp} years in talent acquisition, performance management, and employee relations. "
            "Experience with HRMS systems, payroll processing, and labor law compliance. "
            "Education: MBA Human Resources.",

            "Chartered Accountant with {exp} years in auditing, taxation, and financial reporting. "
            "Expertise in GST, income tax, and IFRS. "
            "Education: CA from ICAI, B.Com.",

            "Civil Engineer with {exp} years in structural design and project management. "
            "Worked on highways, bridges, and residential construction. "
            "Education: B.Tech Civil Engineering. Skills: AutoCAD, STAAD Pro, MS Project.",

            "Graphic Designer with {exp} years creating visual assets for digital and print media. "
            "Expert in Adobe Photoshop, Illustrator, InDesign, and Figma. "
            "Education: B.Des Visual Communication.",

            "Operations Manager with {exp} years in supply chain, logistics, and vendor management. "
            "Experience with ERP systems (SAP, Oracle), procurement, and warehouse operations. "
            "Education: MBA Operations. Skills: Six Sigma, lean manufacturing.",

            "Content Writer with {exp} years producing SEO content, blogs, and technical documentation. "
            "Expertise in WordPress, Grammarly, Hemingway Editor, and AP style. "
            "Education: B.A. English Literature.",

            # ── FIX 3 — HARD NEGATIVE SAMPLES ──────────────────────────
            # These are domain-clear mismatches.  Their keywords share
            # zero overlap with tech JDs, which teaches the model the
            # clearest possible negative boundary.
            "Digital Marketing Specialist with {exp} years running paid media and influencer campaigns. "
            "Managed Google Ads, Facebook Ads Manager, and HubSpot workflows. "
            "Strong skills in A/B testing of ad creatives, email automation, and SEO audits. "
            "Education: B.A. Mass Communication. Background in brand communications and media.",  # FIX: negation phrase removed

            "Mechanical Design Engineer with {exp} years in product development and CAD modeling. "
            "Designed automotive components using SolidWorks and CATIA. "
            "Hands-on experience with FEA simulation, GD&T, and manufacturing tolerances. "
            "Education: B.Tech Mechanical Engineering. Expertise in physical product design.",  # FIX: negation phrase removed

            "Licensed Pharmacist with {exp} years of clinical and retail pharmacy experience. "
            "Expert in drug interactions, patient counseling, and dispensing protocols. "
            "Familiar with pharmacy management systems and insurance billing. "
            "Education: B.Pharm, Pharm.D. Expertise in clinical patient care.",  # FIX: negation phrase removed

            "Primary School Teacher with {exp} years developing K-12 curriculum and classroom management. "
            "Skilled in lesson planning, student assessment, and parent communication. "
            "Education: B.Ed. Expertise in pedagogy and child development.",  # FIX: negation phrase removed

            "Chef and Culinary Arts professional with {exp} years in fine-dining kitchen management. "
            "Expert in menu design, food cost control, and kitchen staff supervision. "
            "Education: Diploma in Culinary Arts. Expertise in hospitality and food service.",  # FIX: negation phrase removed
        ],
    }

    # ── Job description templates per role ──
    jd_templates = {
        "data_scientist": [
            "We are looking for a Data Scientist to join our AI team. "
            "Requirements: 2+ years experience in machine learning, deep learning, and statistical modeling. "
            "Proficiency in Python, scikit-learn, TensorFlow or PyTorch. "
            "Experience with NLP, feature engineering, and model evaluation. "
            "Nice to have: SQL, Spark, cloud platforms (AWS/GCP). "
            "Education: M.Sc. or B.Tech in CS/Statistics/Math.",

            "Hiring a Senior Data Scientist specializing in NLP and LLMs. "
            "Must have: 3+ years ML experience, strong Python, Hugging Face, and generative AI skills. "
            "Experience with MLflow, Docker, and API deployment. "
            "Strong statistical background required. Join our NLP research team.",

            "Data Scientist role focused on recommendation systems and personalization. "
            "Requirements: Python, pandas, scikit-learn, A/B testing experience. "
            "SQL proficiency essential. Experience with data visualization (Tableau/Power BI). "
            "Education: B.Tech or M.Sc. in relevant field.",

            "Data Scientist to improve our fraud detection systems. "
            "Required: Python, XGBoost, scikit-learn, SQL, and strong statistics. "
            "Experience with imbalanced datasets and model interpretability (SHAP/LIME). "
            "Nice to have: Spark, AWS SageMaker. 2-5 years experience.",

            "Junior Data Scientist for our analytics team. "
            "Requirements: Python, pandas, matplotlib, basic machine learning. "
            "SQL required. Exposure to scikit-learn and Jupyter notebooks. "
            "Fresh graduates with strong projects welcome.",
        ],
        "ml_engineer": [
            "ML Engineer needed to build and deploy machine learning systems at scale. "
            "Requirements: Python, TensorFlow/PyTorch, Docker, Kubernetes. "
            "Experience with MLOps tools: MLflow, Airflow, DVC. "
            "Cloud experience on AWS or GCP. CI/CD pipeline knowledge essential. "
            "3+ years experience required.",

            "Senior ML Engineer for computer vision products. "
            "Must have: OpenCV, YOLO/object detection, TensorFlow or PyTorch. "
            "Experience with model optimization (ONNX, quantization, pruning). "
            "Cloud deployment on Azure or AWS. C++ is a plus.",

            "ML Platform Engineer to build internal ML infrastructure. "
            "Requirements: Python, Spark, Kafka, Kubernetes, feature stores. "
            "Experience with Databricks, Snowflake, and real-time prediction APIs. "
            "Strong system design and distributed systems knowledge.",

            "ML Engineer for conversational AI and LLM applications. "
            "Must have: Python, Hugging Face, LangChain, vector databases (Pinecone/Weaviate). "
            "FastAPI and Docker for model serving. GCP or AWS deployment. "
            "2+ years ML engineering experience.",

            "Production ML Engineer to maintain and improve recommendation systems. "
            "Requirements: Python, scikit-learn, Spark MLlib, Kafka, and Redis. "
            "Experience with A/B testing and online learning. Strong SQL. "
            "3+ years in ML production environments.",
        ],
        "data_engineer": [
            "Data Engineer to design and maintain our data infrastructure. "
            "Requirements: Python, SQL, Apache Spark, Airflow, and dbt. "
            "Experience building data pipelines and warehouses on AWS (S3, Redshift) or Snowflake. "
            "3+ years in data engineering. Docker and Kubernetes knowledge preferred.",

            "Senior Data Engineer for real-time streaming platform. "
            "Must have: Kafka, Spark Streaming, PostgreSQL, and Python. "
            "Experience with event-driven architectures and cloud data platforms (GCP/Azure). "
            "Strong SQL skills and understanding of data modeling.",

            "Data Engineer to work on our analytics platform. "
            "Skills required: SQL, Python, ETL development, and dbt. "
            "Familiarity with Airflow and cloud data warehouses. "
            "Junior-to-mid level, 1-3 years experience.",

            "Analytics Engineer to build and maintain our dbt models. "
            "Requirements: dbt, SQL, BigQuery or Snowflake, Looker, and Python. "
            "Experience with data quality, testing, and documentation practices. "
            "Bonus: Airflow, Great Expectations. 2+ years experience.",

            "Data Engineer for our lakehouse modernization project. "
            "Must have: Databricks, Delta Lake, Spark, Python, and SQL. "
            "Terraform for infrastructure. Experience migrating legacy Hadoop workloads. "
            "3-5 years data engineering experience.",
        ],
        "software_engineer": [
            "Backend Software Engineer for our core platform team. "
            "Requirements: Java or Go, microservices, REST APIs, Kafka. "
            "PostgreSQL, Redis, Docker, Kubernetes. CI/CD experience. "
            "3+ years backend development. System design skills required.",

            "Full-Stack Engineer to build our web platform. "
            "Frontend: React, TypeScript. Backend: Node.js or Python/Django. "
            "PostgreSQL, MongoDB, AWS deployment. "
            "2+ years full-stack experience. Git workflow required.",

            "Software Engineer specializing in high-performance systems. "
            "Languages: C++ or Rust, Python scripting. "
            "Distributed systems experience, gRPC, and message queues. "
            "Education: M.Tech or B.Tech CS.",

            "Backend Engineer for fintech payment platform. "
            "Requirements: Java Spring Boot, Kafka, PostgreSQL, Redis. "
            "Strong in security, PCI compliance, and high-throughput system design. "
            "3+ years backend experience. Bonus: Python.",

            "Software Engineer — API Platform team. "
            "Requirements: Python or Go, REST and GraphQL APIs, PostgreSQL. "
            "Docker, Kubernetes, and CI/CD. Good understanding of OAuth and API security. "
            "2-4 years experience. Remote-friendly.",
        ],
        "devops_engineer": [
            "DevOps Engineer for our cloud platform. "
            "Requirements: Kubernetes, Docker, Terraform, Jenkins/GitHub Actions. "
            "AWS or GCP infrastructure management. Monitoring with Prometheus/Grafana. "
            "Strong scripting skills (Python/Bash). 3+ years experience.",

            "SRE to maintain reliability and performance of our infrastructure. "
            "Must have: Kubernetes, Helm, Terraform, Python. "
            "Incident management, SLA/SLO management, and chaos engineering experience. "
            "Strong cloud background (AWS/GCP/Azure).",

            "Platform Engineer to build and improve our internal developer platform. "
            "Requirements: Kubernetes, Terraform, GitHub Actions, ArgoCD, Go or Python. "
            "Experience with Backstage, FinOps, and cost optimization. "
            "Bonus: service mesh (Istio/Linkerd). 4+ years DevOps experience.",
        ],
        "unrelated": [
            "Marketing Specialist to develop and execute digital marketing campaigns. "
            "Requirements: SEO, Google Ads, social media management, content creation. "
            "Experience with CRM tools and market research. MBA preferred.",

            "HR Business Partner for talent acquisition and organizational development. "
            "Requirements: recruitment, performance management, HRMS, labor law knowledge.",

            "Financial Analyst for our accounting and finance team. "
            "Requirements: financial modeling, Excel, accounting principles, GST, tax compliance.",

            "Civil Project Manager for infrastructure projects. "
            "Requirements: AutoCAD, project scheduling, structural design, site supervision.",

            "Content Marketing Manager. Requirements: blog writing, SEO, email campaigns, "
            "HubSpot, Google Analytics. Strong English writing and editorial skills.",

            # ── FIX 3 — HARD NEGATIVE JDs ────────────────────────────
            "Brand Strategy Director for consumer goods company. "
            "Must have: market segmentation, brand architecture, consumer insight research, "
            "campaign planning, and agency management. MBA in Marketing required. "
            "Brand management and consumer research experience essential.",  # FIX: negation phrase removed

            "Structural Engineer for infrastructure consultancy. "
            "Requirements: RCC design, AutoCAD, STAAD Pro, IS code knowledge, "
            "site inspection, and project scheduling. B.Tech Civil Engineering required. "
            "Experience in structural planning and site management required.",  # FIX: negation phrase removed

            "Pediatric Nurse Practitioner for hospital ward. "
            "Requirements: clinical assessment, medication administration, patient care, "
            "EMR documentation, and family counseling. BSN or MSN required. "
            "Clinical certification and patient care experience required.",  # FIX: negation phrase removed
        ],
    }

    samples = []
    tech_roles = ["data_scientist", "ml_engineer", "data_engineer", "software_engineer", "devops_engineer"]

    # ── 1. STRONG MATCHES (label=1): Same role pairs, multiple exp levels + noise repeats ──
    for role in tech_roles:
        r_templates = resume_templates[role]
        j_templates = jd_templates[role]
        for r_tmpl in r_templates:
            for j_tmpl in j_templates:
                for exp in [1, 2, 3, 4, 5, 6]:
                    # Base sample
                    r = r_tmpl.format(exp=exp)
                    samples.append({"resume": r, "job_description": j_tmpl, "label": 1})
                    # Noise variant (slightly randomized text)
                    r_noisy = _noise(r_tmpl, exp)
                    if r_noisy != r:
                        samples.append({"resume": r_noisy, "job_description": j_tmpl, "label": 1})

    # ── 2. UNRELATED PAIRS (label=0): Unrelated resume + tech JD, with noise ──
    unrelated_resumes = resume_templates["unrelated"]
    unrelated_jds     = jd_templates["unrelated"]

    for r_tmpl in unrelated_resumes:
        for exp in [1, 2, 3, 5, 7]:
            r = r_tmpl.format(exp=exp)
            # vs all tech JDs
            for role in tech_roles:
                for j_tmpl in jd_templates[role]:
                    samples.append({"resume": r, "job_description": j_tmpl, "label": 0})
            # vs unrelated JDs (non-tech resume, non-tech JD → 0)
            for j_tmpl in unrelated_jds:
                samples.append({"resume": r, "job_description": j_tmpl, "label": 0})
            # noise repeat
            r_noisy = _noise(r_tmpl, exp)
            if r_noisy != r:
                for role in ["data_scientist", "ml_engineer"]:
                    samples.append({"resume": r_noisy, "job_description": jd_templates[role][0], "label": 0})

    # ── 3. BORDERLINE CASES ──
    # Cross-role tech mismatch (distant roles → label=0)
    for i, role_a in enumerate(tech_roles):
        for j, role_b in enumerate(tech_roles):
            if abs(i - j) >= 2:
                for exp in [1, 2, 3, 4]:
                    r = resume_templates[role_a][0].format(exp=exp)
                    for j_tmpl in jd_templates[role_b][:2]:
                        samples.append({"resume": r, "job_description": j_tmpl, "label": 0})

    # Junior resume vs senior JD → label=0 (borderline)
    junior_ds  = resume_templates["data_scientist"][2].format(exp=1)
    senior_jds = [jd_templates["data_scientist"][0], jd_templates["ml_engineer"][0]]
    for j_tmpl in senior_jds:
        for _ in range(8):
            samples.append({"resume": junior_ds, "job_description": j_tmpl, "label": 0})

    # Adjacent-role cross matches → label=1 (partial fit)
    adjacent_pairs = [
        ("data_scientist", "ml_engineer"),
        ("ml_engineer", "data_scientist"),
        ("data_engineer", "software_engineer"),
        ("software_engineer", "devops_engineer"),
    ]
    for role_a, role_b in adjacent_pairs:
        r = resume_templates[role_a][0].format(exp=3)
        for j_tmpl in jd_templates[role_b][:2]:
            for _ in range(6):
                samples.append({"resume": r, "job_description": j_tmpl, "label": 1})

    # ── 4. REAL-DATA AUGMENTATION PLACEHOLDERS (100–200 extra clean matched rows) ──
    # Add more same-role matched rows with varied experience to reach target size
    for role in tech_roles:
        r_templates = resume_templates[role]
        j_templates = jd_templates[role]
        for r_tmpl in r_templates:
            for j_tmpl in j_templates:
                for exp in range(1, 11):   # 1–10 years
                    r_noisy = _noise(r_tmpl, exp)
                    samples.append({"resume": r_noisy, "job_description": j_tmpl, "label": 1})

    df = pd.DataFrame(samples)
    df = df.drop_duplicates(subset=["resume", "job_description"]).reset_index(drop=True)
    print(f"[INFO] Built synthetic dataset: {len(df)} samples "
          f"| label=1: {(df['label']==1).sum()} | label=0: {(df['label']==0).sum()}")
    return df


def augment_with_real_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Attempt to load and augment with real HuggingFace dataset.
    Falls back gracefully if unavailable.
    """
    if not HF_AVAILABLE:
        print("[INFO] datasets library not available. Skipping real data augmentation.")
        return df

    try:
        print("[INFO] Loading real dataset from HuggingFace for augmentation...")
        # Use a small, publicly available job-posting dataset
        dataset = load_dataset("jacob-hugging-face/job-descriptions",
                               split="train", trust_remote_code=True)
        real_df = dataset.to_pandas()

        # Extract job descriptions
        jd_col = None
        for col in ["job_description", "description", "text", "job_desc"]:
            if col in real_df.columns:
                jd_col = col
                break

        if jd_col is None:
            raise ValueError("No recognizable JD column found in dataset.")

        real_jds = real_df[jd_col].dropna().str.strip()
        real_jds = real_jds[real_jds.str.len() > 100].head(150).tolist()

        # Pair real JDs with our synthetic resumes
        tech_resume = (
            "Experienced Data Scientist with 3 years in machine learning, Python, TensorFlow, "
            "scikit-learn, SQL, and data visualization. M.Sc. Computer Science."
        )
        unrelated_resume = (
            "Marketing Manager with 5 years in brand strategy and campaign management. "
            "MBA Marketing. Skills: SEO, social media, Google Ads."
        )

        new_rows = []
        tech_keywords = ["python", "machine learning", "data", "software", "engineer",
                         "developer", "cloud", "sql", "ai", "analytics"]

        for jd in real_jds:
            jd_lower = jd.lower()
            is_tech = any(kw in jd_lower for kw in tech_keywords)
            if is_tech:
                new_rows.append({"resume": tech_resume, "job_description": jd, "label": 1})
                new_rows.append({"resume": unrelated_resume, "job_description": jd, "label": 0})
            else:
                new_rows.append({"resume": tech_resume, "job_description": jd, "label": 0})

        aug_df = pd.DataFrame(new_rows)
        combined = pd.concat([df, aug_df], ignore_index=True)
        print(f"[INFO] Augmented dataset: {len(df)} synthetic + {len(aug_df)} real = {len(combined)} total samples.")
        return combined.drop_duplicates().reset_index(drop=True)

    except Exception as e:
        print(f"[INFO] Real data augmentation skipped ({e}). Using synthetic dataset only.")
        return df


# ─────────────────────────────────────────────
# SECTION 5: TEXT PREPROCESSING
# ─────────────────────────────────────────────

class TextPreprocessor:
    """Handles all text normalization: lowercase, punctuation removal,
    stopword removal, and lemmatization."""

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words("english"))
        # Keep important tech terms that might be stopwords
        self.stop_words -= {"no", "not", "c", "r"}

    def clean(self, text: str) -> str:
        """Full preprocessing pipeline."""
        if not text or not isinstance(text, str):
            return ""
        # Lowercase
        text = text.lower()
        # Preserve hyphenated terms
        text = re.sub(r"(?<=[a-z])-(?=[a-z])", " ", text)
        # Remove special characters and digits (keep letters & spaces)
        text = re.sub(r"[^a-z\s/#+]", " ", text)
        # Tokenize
        tokens = text.split()
        # Remove stopwords & lemmatize
        # FIX: len > 2 filters single-char residual noise; explicit KEEP_SHORT for tech tokens
        KEEP_SHORT = {'go', 'ai', 'ml', 'dl', 'r', 'c'}
        tokens = [
            self.lemmatizer.lemmatize(t)
            for t in tokens
            if (t not in self.stop_words) and (len(t) > 2 or t in KEEP_SHORT)
        ]
        return " ".join(tokens)


# ─────────────────────────────────────────────
# SECTION 6: RESUME VALIDATOR
# ─────────────────────────────────────────────

class ResumeValidator:
    """Validates that uploaded document is a genuine resume."""

    def validate(self, text: str) -> tuple[bool, str]:
        """Returns (is_valid, message)."""
        if not text or len(text.strip()) < 100:
            return False, "❌ Document is too short or empty. Please upload a valid resume."

        text_lower = text.lower()
        # FIX: Expanded to cover B.Tech, M.Tech, B.E., M.E. degree formats missed before
        extended_keywords = RESUME_REQUIRED_KEYWORDS + [
            "b.tech", "m.tech", "b.e.", "m.e.", "b.sc", "m.sc", "b.s.", "m.s.",
            "phd", "mba", "diploma", "graduated", "professional", "summary"
        ]
        found = [kw for kw in extended_keywords if kw in text_lower]
        # FIX: Raised threshold from 2 to 3 for stricter resume validation
        if len(found) < 3:
            return False, (
                f"❌ This doesn't appear to be a resume. "
                f"Missing key sections. Found only: {found or ['none']}. "
                f"Please upload a document containing education, experience, and skills sections."
            )
        return True, f"✅ Valid resume detected. Found sections: {', '.join(found[:5])}."


# ─────────────────────────────────────────────
# SECTION 7: FILE PARSER
# ─────────────────────────────────────────────

class FileParser:
    """Extracts raw text from PDF and DOCX files."""

    def extract(self, file_path: str) -> str:
        ext = os.path.splitext(file_path)[-1].lower()
        if ext == ".pdf":
            return self._parse_pdf(file_path)
        elif ext in [".docx", ".doc"]:
            return self._parse_docx(file_path)
        else:
            raise ValueError(f"Unsupported file format: {ext}. Please upload PDF or DOCX.")

    def _parse_pdf(self, path: str) -> str:
        text = []
        with open(path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text.append(t)
        return "\n".join(text)

    def _parse_docx(self, path: str) -> str:
        doc = docx.Document(path)
        return "\n".join([p.text for p in doc.paragraphs if p.text.strip()])


# ─────────────────────────────────────────────
# SECTION 8: SKILL EXTRACTOR
# ─────────────────────────────────────────────

class SkillExtractor:
    """
    Extracts skills using:
    1. Canonical skill keyword matching
    2. Synonym mapping
    3. TF-IDF based pseudo-semantic similarity for ranking
    """

    def __init__(self):
        self.canonical_skills = list(SKILL_TAXONOMY.keys())
        self.synonym_map = SYNONYM_MAP

    def _normalize_to_skills(self, text: str) -> set:
        """Map all synonym mentions in text to canonical skill names."""
        text_lower = text.lower()
        found = set()

        # Direct canonical match
        for skill in self.canonical_skills:
            if skill in text_lower:
                found.add(skill)

        # Synonym match
        for synonym, canonical in self.synonym_map.items():
            if synonym in text_lower:
                found.add(canonical)

        return found

    def extract(self, resume_text: str, jd_text: str) -> dict:
        resume_skills = self._normalize_to_skills(resume_text)
        jd_skills = self._normalize_to_skills(jd_text)

        matched = sorted(resume_skills & jd_skills)
        missing = sorted(jd_skills - resume_skills)
        extra = sorted(resume_skills - jd_skills)

        return {
            "resume_skills": sorted(resume_skills),
            "jd_skills": sorted(jd_skills),
            "matched": matched,
            "missing": missing,
            "extra": extra,
        }


# ─────────────────────────────────────────────
# SECTION 9: MODEL TRAINER
# ─────────────────────────────────────────────

class ModelTrainer:
    """
    Trains and evaluates 4 ML models using TF-IDF + StratifiedKFold cross-validation.
    Selects best model by mean F1-score.
    """

    def __init__(self):
        self.preprocessor = TextPreprocessor()
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,        # FIX: was 1 — singletons overfit to noise tokens
            max_df=0.95,     # FIX: ignore terms in >95% of docs (non-discriminative)
        )
        self.models = {}
        self.cv_results = {}
        self.best_model_name = None
        self.best_model = None
        self.best_f1 = 0.0
        self.X_train_tfidf = None
        self.y_train = None
        self.label_classes = None

    def _get_model_definitions(self) -> dict:
        return {
            "Logistic Regression": Pipeline([
                ("clf", LogisticRegression(max_iter=1000, C=1.0, solver="lbfgs", random_state=42))
            ]),
            "Naive Bayes": Pipeline([
                ("clf", MultinomialNB(alpha=0.5))
            ]),
            "SVM": Pipeline([
                ("clf", CalibratedClassifierCV(LinearSVC(C=1.0, max_iter=2000, random_state=42)))
            ]),
            "Random Forest": Pipeline([
                ("clf", RandomForestClassifier(n_estimators=200, max_depth=15,
                                               random_state=42, n_jobs=-1))
            ]),
        }

    def _make_structured_text(self, resume: str, jd: str) -> str:
        """
        FIX 1 — FEATURE SEPARATION
        Instead of a raw concatenation (resume + " " + jd), we prefix each
        section with a repeated sentinel token.  When TF-IDF sees
        'resumesection … jdsection …' it creates distinct n-gram features
        for each region, giving the model a positional signal.

        Example output:
          "resumesection python machine learning … jdsection data scientist …"
        """
        r_clean  = self.preprocessor.clean(resume)
        jd_clean = self.preprocessor.clean(jd)
        # Repeat sentinels so they survive min_df filtering
        return f"resumesection resumesection {r_clean} jdsection jdsection {jd_clean}"

    def train(self, df: pd.DataFrame):
        """Full training pipeline with balanced classes and feature separation."""
        print(f"[INFO] Dataset size: {len(df)} samples | Label distribution:\n{df['label'].value_counts().to_dict()}")

        # ── FIX 2 — DATASET BALANCING ────────────────────────────────
        # Undersample the majority class so the model cannot win by always
        # predicting the dominant label.
        n_pos = (df["label"] == 1).sum()
        n_neg = (df["label"] == 0).sum()
        minority_n = min(n_pos, n_neg)

        df_pos = df[df["label"] == 1].sample(n=minority_n, random_state=42)
        df_neg = df[df["label"] == 0].sample(n=minority_n, random_state=42)
        df_balanced = pd.concat([df_pos, df_neg]).sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"[INFO] Balanced dataset: {len(df_balanced)} samples "
              f"(label=1: {minority_n}, label=0: {minority_n})")

        # ── FIX 1 applied — structured separator for every row ────────
        df_balanced["cleaned"] = df_balanced.apply(
            lambda row: self._make_structured_text(row["resume"], row["job_description"]),
            axis=1
        )

        X_text = df_balanced["cleaned"].values
        y      = df_balanced["label"].values

        # Fit TF-IDF
        X_tfidf = self.vectorizer.fit_transform(X_text)
        self.X_train_tfidf = X_tfidf
        self.y_train = y

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        model_defs = self._get_model_definitions()

        # FIX: Store OOF predictions for honest confusion matrix (not training data)
        self.cv_holdout_preds = None
        self.cv_holdout_true  = None

        print("\n[INFO] Running 5-fold cross-validation for all models...")
        # FIX: First pass — evaluate all models, THEN determine winner (fixes ★ BEST printing)
        for name, pipeline in model_defs.items():
            cv_out = cross_validate(
                pipeline, X_tfidf, y,
                cv=skf,
                scoring="f1",
                return_train_score=True
            )
            mean_f1  = float(np.mean(cv_out["test_score"]))
            std_f1   = float(np.std(cv_out["test_score"]))
            train_f1 = float(np.mean(cv_out["train_score"]))

            self.cv_results[name] = {
                "mean_f1": mean_f1,
                "std_f1": std_f1,
                "train_f1": train_f1,
                "fold_scores": cv_out["test_score"].tolist(),
            }
            print(f"   {name:25s} | F1: {mean_f1:.4f} ± {std_f1:.4f}")

            if mean_f1 > self.best_f1:
                self.best_f1 = mean_f1
                self.best_model_name = name

        # FIX: Print ★ BEST only once, after all models evaluated
        print(f"\n[INFO] Best Model: {self.best_model_name} (F1={self.best_f1:.4f})")

        # Second pass: refit on full data + collect OOF preds for best model
        from sklearn.model_selection import cross_val_predict
        for name, pipeline in model_defs.items():
            if name == self.best_model_name:
                oof_preds = cross_val_predict(pipeline, X_tfidf, y, cv=skf)
                self.cv_holdout_preds = oof_preds
                self.cv_holdout_true  = y
            pipeline.fit(X_tfidf, y)
            self.models[name] = pipeline
            if name == self.best_model_name:
                self.best_model = pipeline

    def get_cv_summary(self) -> pd.DataFrame:
        rows = []
        for name, stats in self.cv_results.items():
            rows.append({
                "Model": name,
                "Mean F1": round(stats["mean_f1"], 4),
                "Std F1": round(stats["std_f1"], 4),
                "Train F1": round(stats["train_f1"], 4),
                "Best Fold F1": round(max(stats["fold_scores"]), 4),
            })
        return pd.DataFrame(rows).sort_values("Mean F1", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
# SECTION 10: RESUME MATCHER
# ─────────────────────────────────────────────

class ResumeMatcher:
    """
    Combines ML probability score + cosine similarity for final matching.
    Formula: Final Score = 0.6 × model_proba + 0.4 × cosine_similarity
    """

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer
        self.preprocessor = TextPreprocessor()
        self.skill_extractor = SkillExtractor()
        self.validator = ResumeValidator()

    def match(self, resume_text: str, jd_text: str) -> dict:
        """
        Full matching pipeline.

        FIX 4 — SCORE CORRECTION
          Old formula: 0.6 × model_proba + 0.4 × cosine
          Problem: model_proba is trained on structured text (with sentinels),
          but at inference we fed raw combined text → distribution mismatch,
          causing inflated probabilities even for unrelated inputs.

          Fix:
            1. Vectorize using the SAME structured format as training.
            2. Penalise the final score when cosine similarity is very low:
               if cosine < 0.05 (near-zero overlap) we cap model_proba at 0.35
               so a totally unrelated pair cannot exceed ~50 %.
            3. Keep the 0.6/0.4 split but on corrected inputs.

        FIX 5 — THRESHOLD LOGIC
          Old threshold: final_pct >= 50  (too lenient)
          New threshold: final_pct >= 55  AND cosine_sim >= 5 %
          This prevents the "cosine=0, model_proba=0.9 → 54 % score → Good Fit"
          failure mode.

        FIX 6 — DEBUG VISIBILITY
          Prints are written so they appear in the Colab / terminal log at
          every inference call.  They show raw components before combination
          so you can trace exactly how the final score was built.
        """

        # ── Step 1: Preprocess ──────────────────────────────────────────
        resume_clean = self.preprocessor.clean(resume_text)
        jd_clean     = self.preprocessor.clean(jd_text)

        # Use the SAME structured separator the model was trained on
        structured_text = (
            f"resumesection resumesection {resume_clean} "
            f"jdsection jdsection {jd_clean}"
        )

        # ── Step 2: Cosine similarity ───────────────────────────────────
        # Computed on the raw cleaned sections (no sentinels) so the angle
        # purely reflects vocabulary overlap between resume and JD.
        try:
            tfidf_matrix = self.trainer.vectorizer.transform([resume_clean, jd_clean])
            cosine_sim   = float(cosine_similarity(tfidf_matrix[0], tfidf_matrix[1])[0][0])
        except Exception as e:
            cosine_sim = 0.0
            print(f'[WARN] Cosine similarity failed: {e}')  # FIX: bare except → except Exception

        # ── Step 3: Model probability ───────────────────────────────────
        # FIX 4a: Transform the structured text (matches training format).
        try:
            structured_vec = self.trainer.vectorizer.transform([structured_text])
            proba          = self.trainer.best_model.predict_proba(structured_vec)[0]
            model_score    = float(proba[1])   # P(Good Fit)
        except Exception as e:
            model_score = 0.0
            print(f'[WARN] Model prediction failed: {e}')  # FIX: bare except → except Exception

        # ── Step 4: Score correction ────────────────────────────────────
        # FIX 4b: If cosine similarity is near-zero the pair is almost
        # certainly unrelated.  Cap model_score so inflated ML confidence
        # cannot push an unrelated pair over the Good-Fit threshold.
        LOW_COSINE_THRESHOLD  = 0.05   # below this → vocabularies share almost nothing
        LOW_COSINE_PROBA_CAP  = 0.35   # cap model_score at 35 % in that case

        cosine_sim_capped = cosine_sim
        if cosine_sim < LOW_COSINE_THRESHOLD:
            model_score = min(model_score, LOW_COSINE_PROBA_CAP)

        # Weighted combination (weights unchanged from original design)
        final_score = 0.6 * model_score + 0.4 * cosine_sim
        final_pct   = round(final_score * 100, 2)

        # ── Step 5: Decision ────────────────────────────────────────────
        # FIX 5: Dual condition — score AND a minimum cosine threshold.
        # A pair cannot be "Good Fit" if the raw vocabulary overlap is
        # near zero, regardless of what the ML model believes.
        SCORE_THRESHOLD  = 55.0   # raised from 50 for tighter boundary
        COSINE_MIN_PCT   = 5.0    # at least 5 % cosine similarity required

        is_good_fit = (final_pct >= SCORE_THRESHOLD) and (cosine_sim * 100 >= COSINE_MIN_PCT)

        # ── Step 6: Debug visibility ────────────────────────────────────
        # FIX 6: Minimal but informative logs — visible in Colab output
        # and in the Gradio terminal.  Useful for viva demonstration.
        print(
            f"\n[MATCH DEBUG] "
            f"cosine_sim={cosine_sim*100:.1f}%  "
            f"model_proba={model_score*100:.1f}%  "
            f"final_score={final_pct:.1f}%  "
            f"verdict={'GOOD FIT' if is_good_fit else 'NOT FIT'}"
        )

        # ── Step 7: Skill analysis & suggestions ───────────────────────
        skill_info  = self.skill_extractor.extract(resume_text, jd_text)
        suggestions = self._generate_suggestions(final_pct, skill_info["missing"], skill_info["matched"])

        return {
            "model_score": round(model_score * 100, 2),
            "cosine_sim":  round(cosine_sim * 100, 2),
            "final_score": final_pct,
            "is_good_fit": is_good_fit,
            "skill_info":  skill_info,
            "suggestions": suggestions,
            "best_model":  self.trainer.best_model_name,
            "best_f1":     round(self.trainer.best_f1, 4),
            "cv_results":  self.trainer.cv_results,
        }

    def _generate_suggestions(self, score: float, missing: list, matched: list) -> list:
        """Generate actionable improvement suggestions."""
        suggestions = []

        if missing:
            top_missing = missing[:5]
            suggestions.append(
                "🎯 Add these missing skills to your resume: **" + ', '.join(top_missing) + "**"  # FIX: clean string concat
            )

        if score < 40:
            suggestions.append(
                "📝 Your resume needs significant alignment with this JD. "
                "Tailor your summary section to mirror the job description language."
            )
        elif score < 60:
            suggestions.append(
                "✏️ Moderate match. Highlight projects where you used the required technologies more explicitly."
            )
        else:
            suggestions.append(
                "✅ Strong match! Quantify your achievements with metrics (e.g., 'Improved accuracy by 15%')."
            )

        if len(matched) < 3:
            suggestions.append(
                "🔍 Use the exact technology names from the JD (e.g., 'TensorFlow' not just 'deep learning frameworks')."
            )

        suggestions.append(
            "📄 Include a 2-3 line professional summary at the top of your resume targeting this specific role."
        )

        if missing:
            suggestions.append(
                f"📚 Consider online certifications for: {', '.join(missing[:3])} "
                f"(Coursera, Udemy, or official documentation)."
            )

        return suggestions[:4]


# ─────────────────────────────────────────────
# SECTION 11: VISUALIZATION ENGINE
# ─────────────────────────────────────────────

class VisualizationEngine:
    """Generates all chart outputs: F1 bar chart, confusion matrix."""

    def __init__(self, trainer: ModelTrainer):
        self.trainer = trainer

    def f1_bar_chart(self) -> plt.Figure:
        """Bar chart comparing all model F1 scores."""
        summary = self.trainer.get_cv_summary()

        fig, ax = plt.subplots(figsize=(9, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")

        colors = ["#00d4ff", "#7c3aed", "#10b981", "#f59e0b"]
        bars = ax.bar(
            summary["Model"],
            summary["Mean F1"],
            color=colors[:len(summary)],
            width=0.5,
            edgecolor="#ffffff20",
            linewidth=0.8,
        )

        # Error bars (std)
        ax.errorbar(
            range(len(summary)), summary["Mean F1"],
            yerr=summary["Std F1"],
            fmt="none", color="white", capsize=5, linewidth=1.5,
        )

        # Value labels
        for bar, val in zip(bars, summary["Mean F1"]):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.015,
                f"{val:.4f}",
                ha="center", va="bottom",
                color="white", fontsize=10, fontweight="bold"
            )

        # Best model highlight
        best_idx = summary["Model"].tolist().index(self.trainer.best_model_name)
        bars[best_idx].set_edgecolor("#ffffff")
        bars[best_idx].set_linewidth(2.5)

        ax.set_ylim(0, 1.12)
        ax.set_ylabel("Mean F1 Score (CV=5)", color="white", fontsize=11)
        ax.set_title("📊 Model Comparison — Cross-Validation F1 Scores", color="white",
                     fontsize=13, fontweight="bold", pad=15)
        ax.tick_params(colors="white")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for spine in ["left", "bottom"]:
            ax.spines[spine].set_color("#ffffff30")
        ax.set_xticklabels(summary["Model"], color="white", fontsize=10)

        best_patch = mpatches.Patch(facecolor="white", edgecolor="white",
                                    label=f"★ Best: {self.trainer.best_model_name}")
        ax.legend(handles=[best_patch], facecolor="#1a1d27",
                  edgecolor="#ffffff30", labelcolor="white", fontsize=9)

        plt.tight_layout()
        return fig

    def confusion_matrix_chart(self) -> plt.Figure:
        """Confusion matrix using out-of-fold predictions (honest held-out performance)."""
        # FIX: Was predicting on TRAINING data → always near-perfect (model memorised it).
        # Now uses OOF predictions = true generalisation estimate.
        if (self.trainer.cv_holdout_preds is not None and
                self.trainer.cv_holdout_true is not None):
            y_pred = self.trainer.cv_holdout_preds
            y_true = self.trainer.cv_holdout_true
        else:
            y_pred = self.trainer.best_model.predict(self.trainer.X_train_tfidf)
            y_true = self.trainer.y_train
        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#1a1d27")

        cmap = sns.diverging_palette(250, 133, s=80, as_cmap=True)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            ax=ax, linewidths=1, linecolor="#ffffff20",
            xticklabels=["Not Fit (0)", "Good Fit (1)"],
            yticklabels=["Not Fit (0)", "Good Fit (1)"],
            annot_kws={"size": 14, "weight": "bold", "color": "white"},
            cbar_kws={"shrink": 0.8},
        )

        ax.set_title(
            f"🧩 Confusion Matrix — {self.trainer.best_model_name}",
            color="white", fontsize=12, fontweight="bold", pad=12
        )
        ax.set_xlabel("Predicted Label", color="white", fontsize=10)
        ax.set_ylabel("True Label", color="white", fontsize=10)
        ax.tick_params(colors="white")
        ax.yaxis.set_tick_params(rotation=0)

        plt.tight_layout()
        return fig

    def score_gauge_chart(self, score: float) -> plt.Figure:
        """Semi-circular gauge showing final match score."""
        fig, ax = plt.subplots(figsize=(6, 3.5), subplot_kw={"polar": False})
        fig.patch.set_facecolor("#0f1117")
        ax.set_facecolor("#0f1117")
        ax.axis("off")

        # Draw gauge arc
        theta = np.linspace(np.pi, 0, 300)
        r_outer, r_inner = 1.0, 0.6

        # Background track
        ax.plot(np.cos(theta), np.sin(theta), color="#ffffff15", linewidth=30)

        # Score fill
        fill_theta = np.linspace(np.pi, np.pi - (score / 100) * np.pi, 300)
        # FIX: Was >=60 for green but Good Fit threshold is 55%. Aligned to match verdict.
        color = "#10b981" if score >= 55 else ("#f59e0b" if score >= 35 else "#ef4444")
        ax.plot(np.cos(fill_theta), np.sin(fill_theta), color=color, linewidth=30)

        # Center text
        ax.text(0, 0.05, f"{score:.1f}%", ha="center", va="center",
                fontsize=28, fontweight="bold", color="white")
        ax.text(0, -0.25, "Match Score", ha="center", va="center",
                fontsize=11, color="#aaaaaa")

        # Labels
        ax.text(-1.15, -0.15, "0%", color="#aaaaaa", fontsize=9)
        ax.text(1.05, -0.15, "100%", color="#aaaaaa", fontsize=9)
        ax.text(0, 1.1, "50%", ha="center", color="#aaaaaa", fontsize=9)

        ax.set_xlim(-1.4, 1.4)
        ax.set_ylim(-0.5, 1.3)
        plt.tight_layout()
        return fig


# ─────────────────────────────────────────────
# SECTION 12: GRADIO UI
# ─────────────────────────────────────────────

class ResumeIntelligencePlatform:
    """Main application class — wires everything together."""

    def __init__(self):
        print("=" * 60)
        print("  RESUME INTELLIGENCE PLATFORM — Initializing...")
        print("=" * 60)

        # Build & augment dataset
        print("\n[STEP 1] Building synthetic dataset...")
        df = build_synthetic_dataset()
        print(f"[INFO] Synthetic: {len(df)} samples")

        print("\n[STEP 2] Augmenting with real data...")
        df = augment_with_real_data(df)
        print(f"[INFO] Final dataset: {len(df)} samples")

        # Train models
        print("\n[STEP 3] Training ML models...")
        self.trainer = ModelTrainer()
        self.trainer.train(df)

        # Build dependent components
        self.matcher = ResumeMatcher(self.trainer)
        self.viz = VisualizationEngine(self.trainer)
        self.parser = FileParser()
        self.validator = ResumeValidator()
        self.preprocessor = TextPreprocessor()

        print("\n[INFO] Platform ready! Launching Gradio UI...\n")

    def process(self, resume_file, jd_text: str):
        """Main handler called by Gradio."""

        # ── Input validation ──
        def warn_html(msg):
            return f'<div style="background:#f59e0b14;border:1px solid #f59e0b30;border-radius:10px;padding:12px 16px;color:#f59e0b;font-size:13px;font-weight:600;">{msg}</div>'
        def err_html(msg):
            return f'<div style="background:#ef444414;border:1px solid #ef444430;border-radius:10px;padding:12px 16px;color:#ef4444;font-size:13px;font-weight:600;">{msg}</div>'

        if resume_file is None:
            return (
                warn_html("⚠️ Please upload a resume file (PDF or DOCX)."),
                "", "", "", "", "", None, None, None, None
            )
        if not jd_text or len(jd_text.strip()) < 100:  # FIX: was 30 — too permissive for meaningful JDs
            return (
                warn_html("⚠️ Please provide a job description (at least 30 characters)."),
                "", "", "", "", "", None, None, None, None
            )

        # ── Parse resume ──
        try:
            resume_text = self.parser.extract(resume_file.name)
        except Exception as e:
            return (err_html(str(e)), "", "", "", "", "", None, None, None, None)

        # ── Validate resume ──
        is_valid, val_msg = self.validator.validate(resume_text)
        if not is_valid:
            return (err_html(val_msg), "", "", "", "", "", None, None, None, None)

        val_html = f'<div style="background:#10b98114;border:1px solid #10b98130;border-radius:10px;padding:12px 16px;color:#10b981;font-size:13px;font-weight:600;">{val_msg}</div>'

        # ── Run matching ──
        result = self.matcher.match(resume_text, jd_text)

        # ── Format outputs ──
        score = result["final_score"]
        is_good_fit = result["is_good_fit"]
        fit_label = "GOOD FIT" if is_good_fit else "NOT A FIT"
        fit_color = "#10b981" if is_good_fit else "#ef4444"
        fit_bg = "#10b98118" if is_good_fit else "#ef444418"
        fit_border = "#10b981" if is_good_fit else "#ef4444"
        fit_icon = "✓" if is_good_fit else "✗"

        # Score fill for arc bar (0–100 mapped to stroke-dasharray)
        arc_pct = score / 100
        circumference = 2 * 3.14159 * 54  # r=54
        stroke_dash = arc_pct * circumference
        stroke_color = "#10b981" if is_good_fit else "#ef4444"

        # ── SCORE CARD (centered, big percentage + verdict) ──
        score_md = f"""
<div style="text-align:center; padding: 28px 20px 20px; background: #13161f; border-radius: 16px; border: 1px solid #ffffff10;">
  <div style="font-size: 13px; font-weight: 600; letter-spacing: 1.5px; text-transform: uppercase; color: #888; margin-bottom: 12px;">MATCH SCORE</div>
  <div style="font-size: 72px; font-weight: 900; color: {stroke_color}; line-height: 1; margin-bottom: 8px;">{score:.0f}<span style="font-size:36px">%</span></div>
  <div style="display: inline-block; margin-top: 10px; padding: 8px 28px; border-radius: 50px; font-size: 16px; font-weight: 800; letter-spacing: 1px; color: {fit_color}; background: {fit_bg}; border: 1.5px solid {fit_border};">
    {fit_icon}&nbsp;&nbsp;{fit_label}
  </div>
  <div style="display: flex; justify-content: center; gap: 24px; margin-top: 20px; flex-wrap: wrap;">
    <div style="text-align:center; background: #1a1d27; border-radius: 10px; padding: 10px 18px; min-width: 110px;">
      <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px;">ML Probability</div>
      <div style="font-size: 22px; font-weight: 800; color: #00d4ff; margin-top: 2px;">{result['model_score']:.1f}%</div>
    </div>
    <div style="text-align:center; background: #1a1d27; border-radius: 10px; padding: 10px 18px; min-width: 110px;">
      <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px;">Cosine Sim</div>
      <div style="font-size: 22px; font-weight: 800; color: #7c3aed; margin-top: 2px;">{result['cosine_sim']:.1f}%</div>
    </div>
    <div style="text-align:center; background: #1a1d27; border-radius: 10px; padding: 10px 18px; min-width: 110px;">
      <div style="font-size: 11px; color: #666; text-transform: uppercase; letter-spacing: 1px;">Final Score</div>
      <div style="font-size: 22px; font-weight: 800; color: {stroke_color}; margin-top: 2px;">{score:.1f}%</div>
    </div>
  </div>
  <div style="margin-top: 14px; font-size: 11px; color: #444;">Formula: 0.6 × ML Probability + 0.4 × Cosine Similarity &nbsp;|&nbsp; Good Fit: score ≥ 55% AND cosine ≥ 5%</div>
</div>
"""

        # ── MODEL INFO CARD ──
        cv = result["cv_results"][result["best_model"]]
        model_md = f"""
<div style="background: #13161f; border-radius: 16px; border: 1px solid #ffffff10; padding: 20px 24px; margin-top: 4px;">
  <div style="font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #666; margin-bottom: 14px;">SELECTED MODEL</div>
  <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 16px;">
    <div style="background: #00d4ff18; border-radius: 8px; padding: 8px 14px; font-size: 14px; font-weight: 700; color: #00d4ff; border: 1px solid #00d4ff30;">{result['best_model']}</div>
    <div style="background: #10b98118; border-radius: 8px; padding: 8px 14px; font-size: 14px; font-weight: 700; color: #10b981; border: 1px solid #10b98130;">F1 {result['best_f1']:.3f}</div>
  </div>
  <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 10px;">
    <div style="background: #1a1d27; border-radius: 8px; padding: 10px 14px;">
      <div style="font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px;">Std Dev</div>
      <div style="font-size: 16px; font-weight: 700; color: #ccc; margin-top: 2px;">±{cv['std_f1']:.4f}</div>
    </div>
    <div style="background: #1a1d27; border-radius: 8px; padding: 10px 14px;">
      <div style="font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px;">Train F1</div>
      <div style="font-size: 16px; font-weight: 700; color: #ccc; margin-top: 2px;">{cv['train_f1']:.4f}</div>
    </div>
  </div>
  <div style="margin-top: 10px; background: #1a1d27; border-radius: 8px; padding: 10px 14px;">
    <div style="font-size: 10px; color: #555; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 4px;">Fold Scores (CV=5)</div>
    <div style="display: flex; gap: 6px; flex-wrap: wrap;">
      {''.join([f'<span style="background:#0f1117; border:1px solid #ffffff15; border-radius:6px; padding:3px 9px; font-size:12px; color:#aaa; font-weight:600;">{round(x,3)}</span>' for x in cv['fold_scores']])}
    </div>
  </div>
</div>
"""

        # ── SKILLS CARD (color-coded tags) ──
        si = result["skill_info"]

        def make_tags(skills, bg, color, border):
            if not skills:
                return '<span style="color:#555; font-size:13px; font-style:italic;">None</span>'
            return " ".join([
                f'<span style="background:{bg}; color:{color}; border:1px solid {border}; '
                f'border-radius:50px; padding:4px 13px; font-size:12px; font-weight:600; '
                f'display:inline-block; margin:3px 2px;">{s}</span>'
                for s in skills
            ])

        matched_tags = make_tags(si["matched"], "#10b98118", "#10b981", "#10b98140")
        missing_tags = make_tags(si["missing"], "#ef444418", "#ef4444", "#ef444440")
        extra_tags   = make_tags(si["extra"][:5], "#3b82f618", "#60a5fa", "#3b82f640")

        skills_md = f"""
<div style="background: #13161f; border-radius: 16px; border: 1px solid #ffffff10; padding: 20px 24px;">
  <div style="font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #666; margin-bottom: 16px;">SKILL ANALYSIS</div>

  <div style="margin-bottom: 16px;">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#10b981;display:inline-block;"></span>
      <span style="font-size:12px; font-weight:700; color:#10b981; text-transform:uppercase; letter-spacing:1px;">Matched&nbsp;({len(si['matched'])})</span>
    </div>
    <div>{matched_tags}</div>
  </div>

  <div style="margin-bottom: 16px;">
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#ef4444;display:inline-block;"></span>
      <span style="font-size:12px; font-weight:700; color:#ef4444; text-transform:uppercase; letter-spacing:1px;">Missing&nbsp;({len(si['missing'])})</span>
    </div>
    <div>{missing_tags}</div>
  </div>

  <div>
    <div style="display:flex; align-items:center; gap:8px; margin-bottom:8px;">
      <span style="width:8px;height:8px;border-radius:50%;background:#60a5fa;display:inline-block;"></span>
      <span style="font-size:12px; font-weight:700; color:#60a5fa; text-transform:uppercase; letter-spacing:1px;">Extra Skills&nbsp;({len(si['extra'])})</span>
    </div>
    <div>{extra_tags}</div>
  </div>
</div>
"""

        # ── SUGGESTIONS ──
        sugg_items = "".join([
            f'<li style="padding: 8px 0; border-bottom: 1px solid #ffffff08; color: #ccc; font-size: 14px; line-height: 1.5;">'
            f'<span style="color:#f59e0b; font-weight:700; margin-right:8px;">→</span>{s}</li>'
            for s in result["suggestions"]
        ])
        sugg_md = f"""
<div style="background: #13161f; border-radius: 16px; border: 1px solid #ffffff10; padding: 20px 24px;">
  <div style="font-size: 11px; font-weight: 700; letter-spacing: 1.5px; text-transform: uppercase; color: #666; margin-bottom: 14px;">💡 IMPROVEMENT SUGGESTIONS</div>
  <ul style="list-style:none; margin:0; padding:0;">{sugg_items}</ul>
</div>
"""

        # ── CV TABLE (Model Insights tab) ──
        cv_df = self.trainer.get_cv_summary()
        cv_table = cv_df.to_markdown(index=False)
        cv_md = f"## 📊 Model Comparison Table\n\n{cv_table}"

        # Charts
        gauge_fig = self.viz.score_gauge_chart(score)
        f1_fig = self.viz.f1_bar_chart()
        cm_fig = self.viz.confusion_matrix_chart()

        return (
            val_html,       # validation status
            score_md,       # score card
            model_md,       # model info
            skills_md,      # skill analysis
            sugg_md,        # suggestions
            cv_md,          # CV comparison table
            gauge_fig,      # gauge chart
            f1_fig,         # F1 bar chart
            cm_fig,         # confusion matrix
        )

    def launch(self):
        """Build and launch the Gradio UI."""

        custom_css = """
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

        * { box-sizing: border-box; }

        /* ── Force dark base ── */
        body, .gradio-container, .main, .wrap {
            font-family: 'Inter', 'Segoe UI', sans-serif !important;
            background: #0f1117 !important;
            color: #e0e0e0 !important;
            max-width: 1100px !important;
            margin: 0 auto !important;
        }

        /* ── Tabs ── */
        .tab-nav {
            background: #13161f !important;
            border-radius: 14px !important;
            padding: 4px !important;
            border: 1px solid #ffffff0d !important;
            gap: 4px !important;
        }
        .tab-nav button {
            color: #aaaaaa !important;
            font-weight: 600 !important;
            font-size: 13px !important;
            border-radius: 10px !important;
            padding: 8px 18px !important;
            background: transparent !important;
            transition: all 0.2s !important;
        }
        .tab-nav button.selected {
            color: #00d4ff !important;
            background: #1e2130 !important;
            border-bottom: none !important;
            box-shadow: 0 0 0 1px #00d4ff30 !important;
        }

        /* ── Panels & blocks ── */
        .block, .panel, .gr-box, .gr-form, .form, .gap, .contain {
            background: #13161f !important;
            border: 1px solid #ffffff0d !important;
            border-radius: 14px !important;
        }

        /* ── Labels ── */
        label span, .label-wrap span, .block > label > span {
            color: #aaaaaa !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            text-transform: uppercase !important;
            letter-spacing: 0.8px !important;
        }

        /* ── Inputs & textareas ── */
        textarea, input[type="text"] {
            background: #0f1117 !important;
            color: #e0e0e0 !important;
            border: 1px solid #ffffff18 !important;
            border-radius: 10px !important;
            font-family: 'Inter', sans-serif !important;
            font-size: 14px !important;
        }
        textarea:focus, input[type="text"]:focus {
            border-color: #00d4ff50 !important;
            box-shadow: 0 0 0 3px #00d4ff10 !important;
        }

        /* ── Markdown / prose: ALL text bright ── */
        .prose, .prose p, .prose li,
        .md, .md p, .md li,
        .markdown-body, .markdown-body p, .markdown-body li {
            color: #e0e0e0 !important;
        }
        .prose h2, .md h2, .markdown-body h2 {
            color: #00d4ff !important;
            font-size: 16px !important;
            margin-top: 0 !important;
            border-bottom: 1px solid #ffffff18 !important;
            padding-bottom: 4px !important;
        }
        .prose h3, .md h3, .markdown-body h3 {
            color: #a78bfa !important;
            font-size: 14px !important;
        }
        strong, b { color: #ffffff !important; }
        em, i { color: #cccccc !important; }

        /* ── Tables ── */
        .prose table, .md table, .markdown-body table, table {
            background: #1a1d27 !important;
            border-collapse: collapse !important;
            width: 100% !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }
        .prose th, .md th, .markdown-body th, th {
            background: #0f1117 !important;
            color: #00d4ff !important;
            font-size: 13px !important;
            font-weight: 700 !important;
            padding: 9px 14px !important;
            border: 1px solid #ffffff15 !important;
        }
        .prose td, .md td, .markdown-body td, td {
            color: #d8d8d8 !important;
            font-size: 13px !important;
            padding: 8px 14px !important;
            border: 1px solid #ffffff10 !important;
        }
        tr:nth-child(even) td { background: #161920 !important; }

        /* ── Code ── */
        .prose code, .md code, .markdown-body code, code {
            background: #0f1117 !important;
            color: #10b981 !important;
            border-radius: 4px;
            padding: 2px 7px;
            font-size: 12px;
            border: 1px solid #ffffff12 !important;
        }

        /* ── Suggestion list ── */
        .prose ol li, .md ol li, .markdown-body ol li,
        .prose ul li, .md ul li, .markdown-body ul li {
            color: #d8d8d8 !important;
            font-size: 14px !important;
            line-height: 1.7 !important;
        }

        /* ── Analyze button ── */
        button.primary {
            background: linear-gradient(135deg, #00c4ee 0%, #7c3aed 100%) !important;
            border: none !important;
            border-radius: 12px !important;
            font-weight: 800 !important;
            font-size: 15px !important;
            color: #fff !important;
            letter-spacing: 0.5px !important;
            padding: 14px 32px !important;
            transition: opacity 0.2s !important;
        }
        button.primary:hover { opacity: 0.88 !important; }

        /* ── File upload ── */
        .upload-container, .file-preview {
            background: #0f1117 !important;
            border: 1.5px dashed #ffffff20 !important;
            border-radius: 12px !important;
            color: #aaaaaa !important;
        }

        /* ── Plot backgrounds ── */
        .plotly-graph-div { background: transparent !important; }
        """

        header_html = """
        <div style="
            background: #13161f;
            border: 1px solid #ffffff0d;
            border-radius: 14px;
            padding: 18px 24px 16px;
            margin-bottom: 4px;
            text-align: center;
        ">
            <h1 style="
                color: #fff; font-size: 22px; font-weight: 800; margin: 0 0 4px;
                letter-spacing: -0.5px;
            ">🧠 Resume <span style="color:#00d4ff;">Intelligence</span> Platform</h1>
            <p style="color: #555; font-size: 13px; margin: 0 0 12px;">
                ML-Powered Resume ↔ Job Description Matching Engine
            </p>
            <div style="display: flex; justify-content: center; gap: 8px; flex-wrap: wrap;">
                <span style="background:#00d4ff12; color:#00d4ff; border:1px solid #00d4ff25; padding:3px 12px; border-radius:50px; font-size:11px; font-weight:700; letter-spacing:0.5px;">TF-IDF + ML</span>
                <span style="background:#7c3aed12; color:#a78bfa; border:1px solid #7c3aed25; padding:3px 12px; border-radius:50px; font-size:11px; font-weight:700; letter-spacing:0.5px;">Skill Gap Analysis</span>
            </div>
        </div>
        """

        with gr.Blocks(css=custom_css, title="Resume Intelligence Platform", theme=gr.themes.Base()) as app:
            gr.HTML(header_html)

            with gr.Tabs():

                # ────────────── TAB 1: INPUT ──────────────
                with gr.TabItem("📥  Input"):
                    gr.HTML("""
                    <div style="padding: 4px 0 12px; color:#666; font-size:13px;">
                        Upload your resume and paste the job description to get an AI-powered match score.
                    </div>
                    """)

                    with gr.Row(equal_height=False):
                        with gr.Column(scale=1):
                            resume_input = gr.File(
                                label="Resume  —  PDF or DOCX",
                                file_types=[".pdf", ".docx"],
                                type="filepath",
                            )
                            status_box = gr.HTML(
                                value='<div style="color:#555; font-size:13px; padding:8px 0;">Upload a resume to begin.</div>'
                            )

                        with gr.Column(scale=1):
                            jd_input = gr.Textbox(
                                label="Job Description",
                                placeholder="Paste the full job description here…\n\nExample: We are looking for a Data Scientist with 2+ years of experience in Python, machine learning, TensorFlow, and SQL…",
                                lines=14,
                                max_lines=28,
                            )

                    analyze_btn = gr.Button(
                        "⚡  Analyze Match",
                        variant="primary",
                        size="lg",
                    )

                # ────────────── TAB 2: RESULTS ──────────────
                with gr.TabItem("📊  Results"):

                    # Row 1: Score (left) + Model info (right)
                    with gr.Row():
                        with gr.Column(scale=1):
                            score_output  = gr.HTML(label="Match Score")
                        with gr.Column(scale=1):
                            model_output  = gr.HTML(label="Model Intelligence")

                    # Row 2: Gauge (left) + Skills (right)
                    with gr.Row():
                        with gr.Column(scale=1):
                            gauge_output  = gr.Plot(label="Score Gauge")
                        with gr.Column(scale=1):
                            skills_output = gr.HTML(label="Skill Analysis")

                    # Row 3: Suggestions full-width
                    sugg_output = gr.HTML(label="Suggestions")

                # ────────────── TAB 3: MODEL INSIGHTS ──────────────
                with gr.TabItem("🔬  Model Insights"):
                    cv_table_output = gr.Markdown(label="Cross-Validation Summary")
                    with gr.Row():
                        f1_chart_output = gr.Plot(label="F1 Score Comparison")
                        cm_chart_output = gr.Plot(label="Confusion Matrix")

            # ── Wire up ──
            analyze_btn.click(
                fn=self.process,
                inputs=[resume_input, jd_input],
                outputs=[
                    status_box,
                    score_output,
                    model_output,
                    skills_output,
                    sugg_output,
                    cv_table_output,
                    gauge_output,
                    f1_chart_output,
                    cm_chart_output,
                ],
            )

        app.launch(debug=True, share=True)


# ─────────────────────────────────────────────
# SECTION 13: ENTRYPOINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    platform = ResumeIntelligencePlatform()
    platform.launch()
