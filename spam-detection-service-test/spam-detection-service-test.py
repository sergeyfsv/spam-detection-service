"""
Данные из ling-spam/messages.csv
Endpoint https://l0id-spam-detection-service.hf.space/predict
"""

import argparse
import pandas as pd
import random
import requests
from typing import Optional
from pydantic import BaseModel, ValidationError
import csv

# Pydantic модель для ответа API
class SpamResponse(BaseModel):
    is_spam: bool
    confidence: float
    message: str

SERVICE_URL = "https://l0id-spam-detection-service.hf.space/predict"


def load_lingspam_data(csv_path: str) -> pd.DataFrame:
    """
    Загружает ling-spam/messages.csv с колонками: subject,message,label
    Обрабатывает кавычки вокруг message и subject.
    """
    df = pd.read_csv(csv_path)

    expected_cols = ['subject', 'message', 'label']
    if not all(col in df.columns for col in expected_cols):
        raise ValueError(f"Ожидаются колонки: {expected_cols}. Найдены: {list(df.columns)}")

    df['subject'] = df['subject'].astype(str).str.strip('"\'')
    df['message'] = df['message'].astype(str).str.strip('"\'')
    
    # Нормализуем label в int
    df['label'] = pd.to_numeric(df['label'], errors='coerce').fillna(0).astype(int)

    print(f"Загружено {len(df)} записей:")
    print(f"  Ham (0): {len(df[df['label'] == 0])}")
    print(f"  Spam (1): {len(df[df['label'] == 1])}")
    
    return df


def sample_data(df: pd.DataFrame, n_samples: int, data_type: str) -> pd.DataFrame:
    """
    Выбирает случайные n_samples записей.
    data_type: 'spam' | 'ham' | 'both'
    """
    if data_type == 'spam':
        sampled = df[df['label'] == 1].sample(n=min(n_samples, len(df[df['label'] == 1])))
    elif data_type == 'ham':
        sampled = df[df['label'] == 0].sample(n=min(n_samples, len(df[df['label'] == 0])))
    elif data_type == 'both':
        total_ham = len(df[df['label'] == 0])
        total_spam = len(df[df['label'] == 1])
        total = total_ham + total_spam
        
        n_ham = int(n_samples * total_ham / total)
        n_spam = n_samples - n_ham
        
        ham_sample = df[df['label'] == 0].sample(n=min(n_ham, total_ham))
        spam_sample = df[df['label'] == 1].sample(n=min(n_spam, total_spam))
        sampled = pd.concat([ham_sample, spam_sample])
    else:
        raise ValueError("data_type должен быть 'spam', 'ham' или 'both'")

    return sampled.sample(frac=1).reset_index(drop=True)  # перемешиваем


def test_email(subject: str, message: str) -> Optional[SpamResponse]:
    payload = {
        "subject": subject,
        "body": message
    }
    
    try:
        response = requests.post(SERVICE_URL, json=payload, timeout=10)
        response.raise_for_status()
        
        result = SpamResponse(**response.json())
        return result
        
    except requests.exceptions.RequestException as e:
        print(f"Ошибка сети: {e}")
        return None
    except ValidationError as e:
        print(f"Ошибка парсинга ответа: {e}")
        return None
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="Тестирование Spam Detection API")
    parser.add_argument("csv_path", help="Путь к messages.csv")
    parser.add_argument("--n", "-n", type=int, default=10, help="Количество тестов (по умолчанию 10)")
    parser.add_argument("--type", "-t", choices=['spam', 'ham', 'both'], 
                       default='both', help="Что тестировать: spam, ham, both")
    
    args = parser.parse_args()

    print(f"естирование {args.n} записей из {args.csv_path}")
    print(f"Тип выборки: {args.type}")

    df = load_lingspam_data(args.csv_path)

    test_data = sample_data(df, args.n, args.type)
    print(f"Выбрано для теста: {len(test_data)} записей")

    results = []
    for idx, row in test_data.iterrows():
        print(f"\nТест #{idx+1}/{len(test_data)} (реальная метка: {'SPAM' if row['label']==1 else 'HAM'})")
        
        result = test_email(row['subject'], row['message'])
        
        if result:
            print(f"\tМодель: {result.message}")
            print(f"\tis_spam: {result.is_spam}, confidence: {result.confidence:.4f}")
            print(f"{'+' if result.is_spam == bool(row['label']) else '-'}")
            
            results.append({
                'true_label': row['label'],
                'pred_is_spam': result.is_spam,
                'confidence': result.confidence,
                'correct': result.is_spam == bool(row['label'])
            })
        else:
            print("\tОшибка API")
            results.append({
                'true_label': row['label'],
                'pred_is_spam': None,
                'confidence': None,
                'correct': False
            })

    print("ИТОГОВАЯ СТАТИСТИКА")
    
    correct = sum(r['correct'] for r in results)
    total = len(results)
    accuracy = correct / total * 100
    
    print(f"Правильных предсказаний: {correct}/{total} ({accuracy:.1f}%)")
    
    if any(r['true_label'] == 1 for r in results):
        spam_correct = sum(r['correct'] for r in results if r['true_label'] == 1)
        spam_total = sum(1 for r in results if r['true_label'] == 1)
        spam_acc = spam_correct / spam_total * 100
        print(f"   Spam precision: {spam_correct}/{spam_total} ({spam_acc:.1f}%)")
    
    if any(r['true_label'] == 0 for r in results):
        ham_correct = sum(r['correct'] for r in results if r['true_label'] == 0)
        ham_total = sum(1 for r in results if r['true_label'] == 0)
        ham_acc = ham_correct / ham_total * 100
        print(f"   Ham precision: {ham_correct}/{ham_total} ({ham_acc:.1f}%)")


if __name__ == "__main__":
    main()
