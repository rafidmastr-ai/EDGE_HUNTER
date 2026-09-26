"""فحص مختصر وحقيقي لاتصال EDGE HUNTER مع Twelve Data.

يقرأ .env بالطريقة نفسها المستخدمة في config/config_hunter.py.
يعرض النتيجة في Terminal فقط ولا يطبع أي مفتاح أو سر.
لا يعدّل قاعدة البيانات أو إعدادات المشروع.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# اجعل جذر المشروع معروفاً عند التشغيل من PyCharm.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# هذا الاستيراد مهم: config_hunter.py هو الذي يحمّل .env في المشروع.
try:
    from config.config_hunter import load_settings
except Exception as exc:
    print("\n=== فحص Twelve Data ===")
    print("النتيجة: خطأ في إعداد المشروع")
    print(f"السبب: تعذر تحميل إعدادات EDGE HUNTER ({type(exc).__name__})")
    raise SystemExit(2)


BASE_URL_DEFAULT = "https://api.twelvedata.com"
SYMBOL = "XAU/USD"
INTERVAL = "5min"


def classify(status: int | None, payload: dict) -> tuple[str, str]:
    code = str(payload.get("code", "")).strip().lower()
    message = str(payload.get("message", "")).strip()
    text = f"{code} {message}".lower()

    # حالات Twelve Data الصريحة أولاً.
    if any(x in text for x in ("trial expired", "trial has expired", "trial_expired")):
        return "انتهت التجربة", "Twelve Data أبلغ صراحةً بانتهاء فترة التجربة."

    if any(x in text for x in ("subscription required", "subscription_required")):
        return "الاشتراك مطلوب", "Twelve Data أبلغ أن الاشتراك مطلوب."

    if status == 429 or code in {"429", "rate_limit"} or "rate limit" in text:
        return "تجاوز الحد", "Twelve Data أبلغ عن تجاوز حد الاستخدام/الطلبات."

    if status in (401, 403) or code in {"401", "403"}:
        return "مفتاح API غير صالح", "Twelve Data رفض بيانات الاعتماد."

    if payload.get("status") == "error" or code:
        return "خطأ من Twelve Data", message or code or "استجابة خطأ."

    return "غير محدد", "وصلت استجابة، لكن لم يمكن تحديد الحالة بدقة."


def main() -> int:
    print("\n=== فحص Twelve Data ===")

    try:
        settings = load_settings()
    except Exception as exc:
        print("النتيجة: خطأ في إعداد المشروع")
        print(f"السبب: تعذر تحميل إعدادات EDGE HUNTER ({type(exc).__name__})")
        return 2

    api_key = (settings.live_provider_api_key or "").strip()
    base_url = (settings.live_provider_url or BASE_URL_DEFAULT).rstrip("/")

    if not api_key:
        print("النتيجة: غير مُهيأ")
        print("السبب: مفتاح Twelve Data غير موجود في إعدادات المشروع.")
        return 2

    # لا يهم هنا أن LIVE_PROVIDER_ENABLED قد يكون false؛
    # هذا الاختبار يفحص صلاحية الاتصال بالحساب نفسه دون تغيير إعدادات التطبيق.
    params = {
        "symbol": SYMBOL,
        "interval": INTERVAL,
        "outputsize": "1",
        "timezone": "UTC",
    }

    request = Request(
        f"{base_url}/time_series?{urlencode(params)}",
        method="GET",
        headers={
            "Accept": "application/json",
            "Authorization": f"apikey {api_key}",
            "User-Agent": "EDGE-HUNTER-TwelveData-Check",
        },
    )

    try:
        with urlopen(request, timeout=settings.live_provider_timeout_seconds) as response:
            status = int(response.status)
            raw = response.read().decode("utf-8", errors="replace")
            payload = json.loads(raw)

        if isinstance(payload, dict) and isinstance(payload.get("values"), list) and payload["values"]:
            print("النتيجة: Twelve Data يعمل ✓")
            print(f"الرمز: {SYMBOL}")
            print(f"الفاصل: {INTERVAL}")
            print("البيانات: تم استلام بيانات حقيقية")
            return 0

        label, reason = classify(status, payload if isinstance(payload, dict) else {})
        print(f"النتيجة: {label}")
        print(f"السبب: {reason}")
        print(f"HTTP: {status}")
        return 1

    except HTTPError as exc:
        try:
            payload = json.loads(exc.read().decode("utf-8", errors="replace"))
        except Exception:
            payload = {}

        label, reason = classify(int(exc.code), payload)
        print(f"النتيجة: {label}")
        print(f"السبب: {reason}")
        print(f"HTTP: {exc.code}")
        print("ملاحظة: لم يتم عرض مفتاح API.")
        return 1

    except (URLError, TimeoutError, OSError) as exc:
        print("النتيجة: خطأ شبكة")
        print("السبب: تعذر الوصول إلى خادم Twelve Data.")
        print(f"التفصيل: {type(exc).__name__}")
        print("ملاحظة: هذه النتيجة لا تثبت انتهاء التجربة أو صلاحية المفتاح.")
        return 1

    except (json.JSONDecodeError, ValueError) as exc:
        print("النتيجة: استجابة غير صالحة")
        print(f"السبب: تعذر قراءة استجابة Twelve Data ({type(exc).__name__})")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
