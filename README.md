# intel_cam

📦 **Intel RealSense D415 + QR-коды**  
Утилита для распознавания QR-кодов на коробках, извлечения параметров и вычисления позы (x,y,z в см + yaw).  
Результаты доступны:
- в файле `box_target.json`
- в UDP-пакетах (JSON) для ноды движения


## Структура проекта
intel_cam/
├── .venv311/              # виртуальное окружение (локально, не в git)
├── box_name.txt           # имя последней распознанной коробки
├── box_target.json        # JSON с позой и yaw
├── diagnose_rs.py         # диагностика камеры
├── grab_frame.py          # сохранение одного кадра
├── qr_preview_target_udp.py   # ✅ основной скрипт: QR + поза + yaw + UDP + визуализация
├── requirements.txt       # зависимости Python
└── README.md              # документация

## Установка окружения (macOS, Python 3.11)
```bash
brew install python@3.11 librealsense cmake pkg-config git zbar

python3.11 -m venv .venv311
source .venv311/bin/activate

pip install --upgrade pip
pip install -r requirements.txt

#запусук Предпросмотр QR и позы + UDP
sudo -E ./.venv311/bin/python qr_preview_target_udp.py \
  --name "Отвертки" \
  --qr_size_cm 4 \
  --udp_host 127.0.0.1 \
  --udp_port 6000 \
  --ready_only \
  --pub_rate_hz 15 \
  --axis_cm 3 \
  --trail 30

	•	--name / --id — фильтрация по коробке (из QR-поля name/id)
	•	--qr_size_cm — реальный размер QR на коробке
	•	--ready_only — публиковать только когда цель стабилизирована
	•	--axis_cm — длина рисуемых осей QR (см)
	•	--trail — длина хвоста центра


##Выходные данные
##Пример JSON (в UDP и в box_target.json):
{
  "stamp": "2025-09-14T09:25:46.289Z",
  "id": "BOX-003",
  "name": "отвертки",
  "pose_cm_smooth": {"x_cm": 3.6, "y_cm": 8.2, "z_cm": 30.2},
  "yaw_deg_smooth": -163.5,
  "ready": true
}

##Доп. утилиты
	•	grab_frame.py — сохранить один кадр с камеры (для отладки)
	•	diagnose_rs.py — проверить подключение камеры


---




👉 Теперь у тебя полный комплект: окружение, зависимостей и документации.  
