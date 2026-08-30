MODEL_GROUPS = [
    {
        "group": "English - General Purpose",
        "models": [
            {"id": "all-MiniLM-L6-v2", "label": "all-MiniLM-L6-v2", "note": "Fastest · 80 MB · recommended default"},
            {"id": "all-MiniLM-L12-v2", "label": "all-MiniLM-L12-v2", "note": "Slightly better than L6, slightly slower"},
            {"id": "all-mpnet-base-v2", "label": "all-mpnet-base-v2", "note": "Best quality in all-* family · 420 MB"},
            {"id": "all-distilroberta-v1", "label": "all-distilroberta-v1", "note": "RoBERTa-based · good balance"},
            {"id": "paraphrase-MiniLM-L6-v2", "label": "paraphrase-MiniLM-L6-v2", "note": "Tuned for paraphrase detection"},
        ]
    },
    {
        "group": "Multilingual",
        "models": [
            {"id": "paraphrase-multilingual-MiniLM-L12-v2", "label": "multilingual-MiniLM-L12-v2", "note": "50+ languages · fast · most popular"},
            {"id": "paraphrase-multilingual-mpnet-base-v2", "label": "multilingual-mpnet-base-v2", "note": "50+ languages · higher quality"},
            {"id": "distiluse-base-multilingual-cased-v2", "label": "distiluse-multilingual-v2", "note": "50+ languages · USE-distilled"},
            {"id": "LaBSE", "label": "LaBSE", "note": "109 languages · best cross-lingual retrieval"},
        ]
    },
    {
        "group": "Retrieval / Asymmetric (Bi-Encoders)",
        "models": [
            {"id": "msmarco-distilbert-base-v4", "label": "msmarco-distilbert-base-v4", "note": "MS MARCO passage retrieval"},
            {"id": "msmarco-MiniLM-L6-v3", "label": "msmarco-MiniLM-L6-v3", "note": "Fast retrieval variant"},
            {"id": "multi-qa-MiniLM-L6-cos-v1", "label": "multi-qa-MiniLM-L6-cos-v1", "note": "Question → answer matching"},
            {"id": "multi-qa-mpnet-base-dot-v1", "label": "multi-qa-mpnet-base-dot-v1", "note": "High quality QA retrieval"},
        ]
    },
    {
        "group": "Instruction-Tuned / Modern (E5 · BGE · GTE)",
        "models": [
            {"id": "intfloat/e5-large-v2", "label": "e5-large-v2", "note": "State-of-the-art on MTEB · use 'query: ' prefix"},
            {"id": "BAAI/bge-large-en-v1.5", "label": "bge-large-en-v1.5", "note": "BGE family · very strong on MTEB"},
            {"id": "BAAI/bge-m3", "label": "bge-m3", "note": "Multilingual BGE · dense + sparse"},
            {"id": "Alibaba-NLP/gte-large-en-v1.5", "label": "gte-large-en-v1.5", "note": "GTE family · competitive on MTEB"},
            {"id": "nomic-ai/nomic-embed-text-v1", "label": "nomic-embed-text-v1", "note": "8192 token context window · open"},
        ]
    },
    {
        "group": "Domain-Specific",
        "models": [
            {"id": "allenai-specter", "label": "specter", "note": "Scientific papers · citation-trained"},
            {"id": "allenai/specter2_base", "label": "specter2", "note": "Scientific papers v2 · improved"},
            {"id": "pritamdeka/S-PubMedBert-MS-MARCO", "label": "S-PubMedBert", "note": "Biomedical text"},
            {"id": "krlvi/sentence-t5-base-nlpl-code", "label": "sentence-t5-code", "note": "Code similarity"},
        ]
    },
]

MODEL_ID_TO_GROUP = {
    m["id"]: g["group"]
    for g in MODEL_GROUPS
    for m in g["models"]
}

DEFAULT_MODEL_ID = "all-MiniLM-L6-v2"

SPACY_LANGUAGES = [
    {"code": "en", "label": "English", "model": "en_core_web_sm"},
    {"code": "de", "label": "German", "model": "de_core_news_sm"},
    {"code": "fr", "label": "French", "model": "fr_core_news_sm"},
    {"code": "es", "label": "Spanish", "model": "es_core_news_sm"},
    {"code": "it", "label": "Italian", "model": "it_core_news_sm"},
    {"code": "pt", "label": "Portuguese", "model": "pt_core_news_sm"},
    {"code": "nl", "label": "Dutch", "model": "nl_core_news_sm"},
    {"code": "pl", "label": "Polish", "model": "pl_core_news_sm"},
    {"code": "ru", "label": "Russian", "model": "ru_core_news_sm"},
    {"code": "zh", "label": "Chinese", "model": "zh_core_web_sm"},
    {"code": "ja", "label": "Japanese", "model": "ja_core_news_sm"},
    {"code": "ko", "label": "Korean", "model": "ko_core_news_sm"},
    {"code": "ar", "label": "Arabic", "model": "ar_core_news_sm"},
    {"code": "ca", "label": "Catalan", "model": "ca_core_news_sm"},
    {"code": "da", "label": "Danish", "model": "da_core_news_sm"},
    {"code": "fi", "label": "Finnish", "model": "fi_core_news_sm"},
    {"code": "el", "label": "Greek", "model": "el_core_news_sm"},
    {"code": "hr", "label": "Croatian", "model": "hr_core_news_sm"},
    {"code": "lt", "label": "Lithuanian", "model": "lt_core_news_sm"},
    {"code": "mk", "label": "Macedonian", "model": "mk_core_news_sm"},
    {"code": "nb", "label": "Norwegian", "model": "nb_core_news_sm"},
    {"code": "ro", "label": "Romanian", "model": "ro_core_news_sm"},
    {"code": "sl", "label": "Slovenian", "model": "sl_core_news_sm"},
    {"code": "sv", "label": "Swedish", "model": "sv_core_news_sm"},
    {"code": "uk", "label": "Ukrainian", "model": "uk_core_news_sm"},
]
