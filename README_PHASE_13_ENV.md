# Phase 13 — API Key / `.env` setup

ضع نسخة من `.env.example` باسم `.env` في جذر المشروع:

```text
EDGE_HUNTER_STRUCTURE/.env
```

ثم افتح الملف وضع مفتاح Twelve Data مكان:

```env
EDGE_HUNTER_LIVE_PROVIDER_API_KEY=ضع مفتاحك هنا
```

ولا ترسل المفتاح داخل المحادثة.

## التثبيت

داخل بيئة المشروع `.venv`:

```powershell
.\.venv\Scripts\python.exe -m pip install "python-dotenv>=1.0,<2.0"
```

بعدها يتحمل `config_hunter.py` ملف `.env` تلقائيًا عند بدء التطبيق، مع عدم استبدال متغيرات البيئة الموجودة مسبقًا.

## التحقق

```powershell
.\.venv\Scripts\python.exe -c "from config.config_hunter import load_settings; s=load_settings(); print('LIVE KEY CONFIGURED:', bool(s.live_provider_api_key)); print('PROVIDER:', s.live_provider_name)"
```

المفترض بعد وضع المفتاح:

```text
LIVE KEY CONFIGURED: True
PROVIDER: twelvedata
```

لا يتم طباعة قيمة المفتاح نفسها.
