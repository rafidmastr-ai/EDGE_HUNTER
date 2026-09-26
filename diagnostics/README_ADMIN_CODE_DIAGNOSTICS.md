# تشخيص تلقائي لتوليد أكواد الإدارة

عند فشل `POST /api/admin/codes` يحفظ EDGE HUNTER التقرير تلقائيًا داخل مجلد `logs/` دون الحاجة إلى تنفيذ أي أمر PowerShell.

الملفات الناتجة:

- `logs/admin_code_generation_errors.jsonl` — سجل تراكمي لكل فشل.
- `logs/admin_code_generation_latest.json` — أحدث فشل فقط، لسهولة رفعه أو مشاركته للتشخيص.

التقارير لا تتضمن كلمة المرور أو Session/Cookie tokens أو مفاتيح API أو النص الخام لكود الاشتراك.
