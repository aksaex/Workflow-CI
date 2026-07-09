# Menggunakan Python 3.10 yang stabil
FROM python:3.10-slim

# Menentukan folder kerja di dalam Docker
WORKDIR /app

# Memindahkan semua file kodingan ke dalam Docker
COPY . /app

# Menginstal library yang dibutuhkan secara manual
RUN pip install pandas scikit-learn mlflow==2.19.0 dagshub matplotlib seaborn virtualenv

# Perintah otomatis saat Docker dijalankan
CMD ["python", "modelling_tuning.py"]