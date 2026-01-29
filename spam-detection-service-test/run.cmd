.\spam-detection-service-test\activate.bat
pip install --no-cache-dir --upgrade -r requirements.txt

python spam-detection-service-test.py .\data\ling-spam\messages.csv --n 100 --type spam

python spam-detection-service-test.py .\data\ling-spam\messages.csv --n 100 --type both

python spam-detection-service-test.py .\data\ling-spam\messages.csv --n 100 --type ham
