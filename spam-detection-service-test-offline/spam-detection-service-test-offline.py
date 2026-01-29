import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sentence_transformers import SentenceTransformer
import xgboost as xgb
from bs4 import BeautifulSoup

# Копируем код загрузки модели из spam-detection-service.py
def load_models(model_path: str = "./../spam-detection-train-workflow/model_artifacts/model.json", threshold_path: str = "./../spam-detection-train-workflow/model_artifacts/threshold.txt"):
    """
    Загружает модели точно так же, как в spam-detection-service.py
    """
    print("Загрузка BERT модели...")
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
    
    print("Загрузка XGBoost модели...")
    model_xgb = xgb.XGBClassifier()
    model_xgb.load_model(model_path)
    
    # Загружаем порог (необязательно для локального тестирования)
    try:
        with open(threshold_path, "r") as f:
            THRESHOLD = float(f.read().strip())
        print(f"Порог загружен: {THRESHOLD}")
    except:
        THRESHOLD = 0.5
        print("Порог не найден, используется 0.5")
    
    return model_xgb, embedder, THRESHOLD

def clean_text(subject: str, body: str) -> str:
    """
    Точная копия функции clean_text из spam-detection-service.py
    """
    if "<html" in body.lower() or "<div" in body.lower():
        soup = BeautifulSoup(body, "html.parser")
        body = soup.get_text(separator=' ')
    
    full_text = f"{subject} . {body}"
    return " ".join(full_text.split())


def load_lingspam_data(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Проверяем колонки
    expected_cols = ['subject', 'message', 'label']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Ожидаются колонки: {expected_cols}")

    # Убираем кавычки из subject и message
    df['subject'] = df['subject'].astype(str).str.strip('"\'')
    df['message'] = df['message'].astype(str).str.strip('"\'')
    
    # Нормализуем label
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    print(f"Загружено {len(df)} записей:")
    print(f"   Ham (0): {len(df[df['label'] == 0])}")
    print(f"   Spam (1): {len(df[df['label'] == 1])}")
    
    return df


def evaluate_model(df: pd.DataFrame, model_xgb, embedder, threshold: float):
    """
    Тестирует модель на всём датасете и выводит метрики.
    """
    print("\nГенерация эмбеддингов...")
    
    # Очищаем текст
    df['clean_text'] = df.apply(lambda row: clean_text(row['subject'], row['message']), axis=1)
    
    # Фильтруем пустые тексты
    df = df[df['clean_text'].str.len() > 15].reset_index(drop=True)
    
    texts = df['clean_text'].tolist()
    
    # Векторизация батчами
    embeddings = embedder.encode(texts, batch_size=32, show_progress_bar=True)
    
    print("\nПредсказания...")
    # Получаем вероятности
    probas = model_xgb.predict_proba(embeddings)
    spam_probs = probas[:, 1]  # вероятность спама
    
    # Применяем порог
    predictions = (spam_probs >= threshold).astype(int)
    
    true_labels = df['label'].values
    
    # Метрики
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions, zero_division=0)
    recall = recall_score(true_labels, predictions, zero_division=0)
    f1 = f1_score(true_labels, predictions, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'spam_probs': spam_probs,
        'predictions': predictions,
        'true_labels': true_labels
    }


def print_results(results):
    print("\n" + "="*60)
    print("РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ НА LING-SPAM")
    print("="*60)
    
    print(f"Accuracy:     {results['accuracy']:.4f} ({results['accuracy']*100:.1f}%)")
    print(f"Precision:    {results['precision']:.4f} ({results['precision']*100:.1f}%)")
    print(f"Recall:       {results['recall']:.4f} ({results['recall']*100:.1f}%)")
    print(f"F1-Score:    {results['f1']:.4f} ({results['f1']*100:.1f}%)")
    
    print("\nConfusion Matrix:")
    print("           Predicted")
    print("           Ham    Spam")
    print(f"Ham       {results['confusion_matrix'][0,0]:5d}  {results['confusion_matrix'][0,1]:5d}")
    print(f"Spam      {results['confusion_matrix'][1,0]:5d}  {results['confusion_matrix'][1,1]:5d}")
    
    # Детали по классам
    n_ham = np.sum(results['true_labels'] == 0)
    n_spam = np.sum(results['true_labels'] == 1)
    print(f"\nРаспределение в датасете:")
    print(f"   Ham:  {n_ham} ({n_ham/len(results['true_labels'])*100:.1f}%)")
    print(f"   Spam: {n_spam} ({n_spam/len(results['true_labels'])*100:.1f}%)")


def main():
    if not os.path.exists("./../spam-detection-train-workflow/model_artifacts/model.json"):
        print("Файл model_artifacts/model.json не найден!")
        print("Сначала запустите обучение модели!")
        return

    model_xgb, embedder, threshold = load_models()

    csv_path = r".\data\ling-spam\messages.csv"
    if not os.path.exists(csv_path):
        print(f"Файл {csv_path} не найден!")
        return

    df = load_lingspam_data(csv_path)

    results = evaluate_model(df, model_xgb, embedder, threshold)

    print_results(results)


if __name__ == "__main__":
    main()
