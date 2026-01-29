import os
import glob
import numpy as np
import pandas as pd
import email
from email.policy import default
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
import re

# ==========================================
# 1. НОРМАЛИЗАЦИЯ И ЗАГРУЗКА ДАННЫХ
# ==========================================

# === spam assassin ===

def extract_body_rfc822(raw: str) -> str:
    """
    Извлекает ТОЛЬКО тело письма по RFC:
    - отбрасывает mbox-строку 'From ...' если есть;
    - отбрасывает все заголовки до первой пустой строки;
    - возвращает текст тела.
    """
    # 1. Убираем mbox-строку, если она есть
    lines = raw.splitlines()
    if lines and lines[0].startswith("From "):
        lines = lines[1:]

    # 2. Находим первую пустую строку (разделитель header/body)
    body_start_idx = 0
    for i, line in enumerate(lines):
        if line.strip() == "":
            body_start_idx = i + 1
            break

    body_lines = lines[body_start_idx:]
    body = "\n".join(body_lines)

    return body

PGP_BEGIN = "-----BEGIN PGP SIGNATURE-----"
PGP_END = "-----END PGP SIGNATURE-----"

def strip_pgp_signatures(text: str) -> str:
    """
    Remove inline PGP signatures:
    -----BEGIN PGP SIGNATURE-----
    ...
    -----END PGP SIGNATURE-----
    """
    pattern = re.compile(
        re.escape(PGP_BEGIN) + r".*?" + re.escape(PGP_END),
        flags=re.DOTALL
    )
    return re.sub(pattern, "", text)

def strip_mime_boundaries(text: str) -> str:
    """
    Remove MIME boundary lines like:
      --==_Exmh_-763629846P
      --boundary_12345--
    General rule: lines starting with two dashes and containing
    mostly non-word / boundary-like tokens.
    """
    cleaned_lines = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("--") and not stripped.lstrip("-").strip():
            # line is only dashes -> skip
            continue
        # typical boundary pattern: starts with -- and has no spaces
        if stripped.startswith("--") and " " not in stripped:
            continue
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)

def strip_mailing_list_footer(text: str) -> str:
    """
    Remove common mailing list footers like:
    _______________________________________________
    Exmh-workers mailing list
    ...
    listinfo/..., mailman/listinfo, etc.
    Strategy: find a 'separator' line and drop everything below it.
    """
    lines = text.splitlines()

    cut_idx = len(lines)
    for i, line in enumerate(lines):
        stripped = line.strip()
        # separator line (e.g. 40+ underscores)
        if len(stripped) >= 10 and set(stripped) == {"_"}:
            cut_idx = i
            break
        # heuristic: “mailing list” marker
        if "mailing list" in stripped.lower():
            cut_idx = i
            break

    lines = lines[:cut_idx]

    # additionally drop trailing 1–2 lines with 'listinfo' links
    while lines and ("listinfo" in lines[-1].lower() or "mailman" in lines[-1].lower()):
        lines.pop()

    return "\n".join(lines)

URL_PATTERN = re.compile(
    r"""(
        https?://\S+      |   # http://... или https://...
        www\.\S+          |   # www.example.com/...
        [\w\.-]+\.[a-z]{2,}(?:/[^\s]*)?  # голые домены типа example.com, foo.co.uk, bar.io/path
    )""",
    re.IGNORECASE | re.VERBOSE,
)

def strip_urls_and_domains(text: str) -> str:
    # полностью удаляем URL/домены
    return re.sub(URL_PATTERN, " ", text)

# --- Обрезка концовки по типичным фразам (подписи) ---

SIGNOFF_MARKERS = [
    "best regards",
    "kind regards",
    "regards,",
    "regards",
    "sincerely",
    "cheers",
    "yours truly",
    "yours faithfully",
    "thank you,",
    "thank you.",
    "thanks,",
    "thanks.",
]

def strip_trailing_signature(text: str) -> str:
    """
    Обрезает текст по типичным фразам окончания письма.
    Всё, что после маркера (включая строку с маркером) — выбрасывается.
    """
    lines = text.splitlines()
    cut_idx = len(lines)
    for i, line in enumerate(lines):
        lower = line.strip().lower()
        for marker in SIGNOFF_MARKERS:
            if marker in lower:
                cut_idx = i
                break
        if cut_idx != len(lines):
            break

    return "\n".join(lines[:cut_idx])

def clean_spamassassin_email_body(raw_content: str) -> str:
    """
    Берем только тело письма (без заголовков) и,
    при необходимости, чистим HTML.
    """
    body = extract_body_rfc822(raw_content)

    # PGP signatures
    body = strip_pgp_signatures(body)

    # MIME boundaries
    body = strip_mime_boundaries(body)

    # mailing list footers
    body = strip_mailing_list_footer(body)

    # strip trailing signatures (Best regards / Kind regards / etc.)
    body = strip_trailing_signature(body)

    # remove URLs / domains
    body = strip_urls_and_domains(body)

    # HTML cleanup if needed
    if "<html" in body.lower() or "<body" in body.lower() or "<div" in body.lower():
        soup = BeautifulSoup(body, "html.parser")
        body = soup.get_text(separator=" ")

    # Нормализуем пробелы/переносы
    body = " ".join(body.split())
    return body

def load_spamassassin_folder_body_only(path: str, label: int):
    """
    Загружает все файлы из папки SpamAssassin, извлекает ТОЛЬКО тело писем.
    label: 1 для spam, 0 для ham.
    """
    texts, labels = [], []

    files = glob.glob(os.path.join(path, "**/*"), recursive=True)
    for fp in files:
        if not os.path.isfile(fp):
            continue
        try:
            with open(fp, "r", encoding="latin-1", errors="ignore") as f:
                raw = f.read()
            clean_txt = clean_spamassassin_email_body(raw)
            if len(clean_txt) < 15:
                continue
            texts.append(clean_txt)
            labels.append(label)
        except Exception as e:
            print(f"Exception of parsing spamassasin email: {e}")

    return texts, labels

def parse_spamassassin_corpus(ham_dir: str, spam_dir: str) -> pd.DataFrame:
    """
    Парсер SpamAssassin, который:
    - полностью игнорирует все заголовки;
    - берет только body по первой пустой строке.
    """
    ham_texts, ham_labels = load_spamassassin_folder_body_only(ham_dir, 0)
    spam_texts, spam_labels = load_spamassassin_folder_body_only(spam_dir, 1)

    texts = ham_texts + spam_texts
    labels = ham_labels + spam_labels

    return pd.DataFrame({"text": texts, "label": labels})

# === Обработка enron ===

def clean_enron_email_text(raw_text: str) -> str:
    """
    Очищает одно письмо Enron из emails_ham_spam.csv по правилам:
    - Убирает строки начиная с:
      '---' (или ' - - - - - original message - - - - - ' и подобные),
      'from:', 'sent:', 'to:', 'subject:' (без учета регистра).
    - Сохраняет остальной текст.
    """
    # Нормализуем переводы строк
    lines = raw_text.splitlines()

    cleaned_lines = []
    for line in lines:
        stripped = line.lstrip()  # без начальных пробелов для проверки

        # Признак "оригинального сообщения" (пересланные блоки)
        if stripped.startswith('---'):
            continue
        # Дополнительно уберём " - - - - - original message - - - - - " и вариации
        if re.match(r"^-+\s*original message\s*-+", stripped, flags=re.IGNORECASE):
            continue

        # Убираем служебные заголовки (from/sent/to/subject)
        lowered = stripped.lower()
        if lowered.startswith('from:'):
            continue
        if lowered.startswith('sent:'):
            continue
        if lowered.startswith('to:'):
            continue
        if lowered.startswith('subject:'):
            continue

        cleaned_lines.append(line)

    # Склеиваем обратно
    cleaned_text = "\n".join(cleaned_lines).strip()
    return cleaned_text

def parse_enron_csv(path_csv: str) -> pd.DataFrame:
    """
    Парсер для Enron в формате emails_ham_spam.csv.
    Ожидается CSV с колонками: label,email_text.
    Возвращает DataFrame с колонками: text, label (0/1).
    """
    df = pd.read_csv(path_csv)

    # Проверим наличие нужных колонок
    assert 'label' in df.columns and 'origin' in df.columns, \
        "Ожидаются колонки 'label' и 'origin' в CSV."

    # Очистка текстов
    df['text'] = df['origin'].astype(str).apply(clean_enron_email_text)

    def normalize_label(x):
        if isinstance(x, str):
            x_low = x.lower().strip()
            if x_low in ('spam', '1'):
                return 1
            if x_low in ('ham', '0'):
                return 0
        return int(x)

    df['label'] = df['label'].apply(normalize_label)

    df_result = df[['text', 'label']].copy()
    return df_result

def build_combined_dataset(enron_csv_path: str, sa_ham_dir: str, sa_spam_dir: str) -> pd.DataFrame:
    """
    Готовит единый DataFrame из двух источников:
    - Enron CSV (emails_ham_spam.csv)
    - SpamAssassin (ham_dir, spam_dir)
    """
    df_enron = parse_enron_csv(enron_csv_path)
    df_sa = parse_spamassassin_corpus(sa_ham_dir, sa_spam_dir)

    df_all = pd.concat([df_enron, df_sa], ignore_index=True)
    # Перемешаем
    df_all = df_all.sample(frac=1, random_state=42).reset_index(drop=True)
    return df_all

def prepare_dataset():
    df = build_combined_dataset(
        enron_csv_path="data/enron/emails_ham_spam.csv",
        sa_ham_dir="data/spamassassin/easy_ham",
        sa_spam_dir="data/spamassassin/spam"
    )


    print(f"\n[INFO] Итоговый датасет: {len(df)} строк.")
    print(f"   -> Ham (0): {sum(df['label'] == 0)}")
    print(f"   -> Spam (1): {sum(df['label'] == 1)}")

    return df

# ==========================================
# 2. ВЕКТОРИЗАЦИЯ (TRANSFORMERS)
# ==========================================

def get_embeddings(texts):
    print("\n[INFO] Генерация эмбеддингов (SentenceTransformer)...")
    # Модель 'all-MiniLM-L6-v2' (384 мерность, быстрая, учитывает контекст)
    model = SentenceTransformer('all-MiniLM-L6-v2')

    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)
    return embeddings

# ==========================================
# 3. ОБУЧЕНИЕ МОДЕЛИ (XGBOOST)
# ==========================================

def train_and_optimize(X, y):
    print("\n[INFO] Старт кросс-валидации (5 фолдов)...")

    # Классификатор
    # scale_pos_weight=1, так как мы будем регулировать threshold
    clf = xgb.XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=42,
        n_jobs=-1
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Получаем вероятности спама (Class 1) для всех данных
    y_probas = cross_val_predict(clf, X, y, cv=cv, method='predict_proba')
    spam_probs = y_probas[:, 1]

    # --- Подбор порога (Threshold Tuning) ---
    print("\n[INFO] Подбор порога для минимизации False Positives:")
    print(f"{'Threshold':<10} | {'Precision':<10} | {'Recall':<10} | {'F1':<10} | {'FP Count':<10}")
    print("-" * 65)

    best_thresh = 0.5
    best_score = 0
    target_prec = 0.98 # Желаемая точность 98%+

    # Сканируем пороги
    thresholds = np.arange(0.5, 0.99, 0.05)

    for t in thresholds:
        y_pred = (spam_probs >= t).astype(int)

        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)

        # Считаем реальное количество False Positives
        # FP = (Pred=1, Real=0)
        fp_count = np.sum((y_pred == 1) & (y == 0))

        print(f"{t:<10.2f} | {prec:.4f}     | {rec:.4f}     | {f1:.4f}     | {fp_count}")

        # Эвристика выбора лучшего порога:
        # 1. Precision должен быть максимально высоким (в идеале > 0.98)
        # 2. При выполнении п.1, максимизируем F1

        if prec >= target_prec:
            if f1 > best_score:
                best_score = f1
                best_thresh = t
        # Если ни один порог не дает target_prec, ищем просто лучший precision
        elif best_score == 0 and prec > 0.9:
             # Временное решение, если модель слабая
             best_thresh = t

    # Если совсем всё плохо и best_thresh не обновился, берем дефолт 0.5, но лучше самый строгий
    if best_score == 0:
        best_thresh = thresholds[np.argmax([precision_score(y, (spam_probs >= t).astype(int)) for t in thresholds])]

    print(f"\n[RESULT] Выбран оптимальный порог: {best_thresh}")

    # Финальные метрики
    final_preds = (spam_probs >= best_thresh).astype(int)
    print("\nОтчет классификации:")
    print(classification_report(y, final_preds, target_names=['Ham', 'Spam']))

    # Обучение финальной модели на всем датасете
    print("[INFO] Обучение финальной модели...")
    clf.fit(X, y)

    return clf, best_thresh

# ==========================================
# MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    # Проверка на наличие папок
    if not os.path.exists('./data'):
        print("[ERROR] Папка ./data не найдена.")
        exit()

    # Загрузка
    df = prepare_dataset()

    # Векторизация
    X_embeddings = get_embeddings(df['text'].tolist())

    # Обучение
    model, threshold = train_and_optimize(X_embeddings, df['label'].values)

    print(f"Модель готова. Используйте порог {threshold} для классификации.")

    # Сохраняем модель XGBoost
    os.makedirs("model_artifacts", exist_ok=True)
    model.save_model("model_artifacts/model.json")

    # Сохраняем пороги классификации
    with open("model_artifacts/threshold.txt", "w") as f:
        f.write(str(threshold))

    print("Модель и пороги классификации сохранены в 'model_artifacts/'")
