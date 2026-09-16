#!/usr/bin/env python3
"""
邮件推送脚本 — 从 GitHub Actions workflow 抽取，本地对齐。
读取 .env 中的 SMTP 配置，把 output/unified_*.html + .json 发到 EMAIL_TO。
3 次重试，QQ 邮箱用授权码。
"""
import os, smtplib, glob, sys, time
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email import encoders
from email.header import Header
from email.utils import formataddr, parseaddr
from datetime import datetime


def load_env(path=".env"):
    """简易 .env 加载（无需 python-dotenv）"""
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


def main():
    load_env()
    date_str = datetime.now().strftime("%Y-%m-%d")
    host = os.environ.get("SMTP_HOST", "")
    port = int(os.environ.get("SMTP_PORT", "465"))
    user = os.environ.get("SMTP_USER", "")
    passwd = os.environ.get("SMTP_PASS", "")
    to = os.environ.get("EMAIL_TO", "")

    if not all([host, user, passwd, to]):
        print("❌ 邮件配置缺失，请检查 .env: SMTP_HOST/SMTP_PORT/SMTP_USER/SMTP_PASS/EMAIL_TO")
        sys.exit(1)

    # SMTP 协议要求 user/passwd 必须为 ASCII（非 ASCII 字符会让 smtplib.login 抛
    # 'ascii' codec 错误）。QQ 邮箱授权码是 16 位 ASCII，.env 占位符为中文会触发此问题。
    for label, val in [("SMTP_USER", user), ("SMTP_PASS", passwd), ("EMAIL_TO", to)]:
        try:
            val.encode("ascii")
        except UnicodeEncodeError:
            print(f"❌ {label} 含非 ASCII 字符：{val!r}")
            print(f"   若是授权码占位符，请到 QQ 邮箱→设置→账户→SMTP 服务生成 16 位授权码填入 .env")
            sys.exit(1)

    html_files = sorted(glob.glob("output/unified_*.html"))
    if not html_files:
        print("❌ 未找到报告文件 output/unified_*.html")
        sys.exit(1)

    html_file = html_files[-1]
    json_files = sorted(glob.glob("output/unified_*.json"))

    with open(html_file, "r", encoding="utf-8") as f:
        html_content = f.read()

    msg = MIMEMultipart("alternative")
    # 用 formataddr 包装 From/To，中文显示名不丢；用 Header 包装 Subject 防 ascii codec 报错
    from_display = formataddr((str(Header("A股统一评分", "utf-8")), user))
    to_display = formataddr((str(Header("左戈", "utf-8")), to))
    msg["From"] = from_display
    msg["To"] = to_display
    period = "盘前" if datetime.now().hour < 11 else "午盘"
    msg["Subject"] = Header(f"统一评分日报 {date_str}（{period}更新）", "utf-8").encode()

    # HTML 正文（超 300K 截断到安全闭合点）
    if len(html_content) > 300000:
        cut = html_content[:300000]
        for tag in ["</tbody></table>", "</table>", "</div>"]:
            idx = cut.rfind(tag)
            if idx > 0:
                cut = cut[: idx + len(tag)]
                break
        html_content_body = (
            cut
            + '<p style="text-align:center;color:#999;padding:20px">完整报告见附件</p>'
        )
    else:
        html_content_body = html_content

    msg.attach(MIMEText(html_content_body, "html", "utf-8"))

    # 附件：完整 HTML
    with open(html_file, "rb") as f:
        part = MIMEBase("application", "octet-stream")
        part.set_payload(f.read())
        encoders.encode_base64(part)
        part.add_header("Content-Disposition", f"attachment; filename=unified_{date_str}.html")
        msg.attach(part)

    # 附件：JSON
    if json_files:
        with open(json_files[-1], "rb") as f:
            part = MIMEBase("application", "octet-stream")
            part.set_payload(f.read())
            encoders.encode_base64(part)
            part.add_header("Content-Disposition", f"attachment; filename=unified_{date_str}.json")
            msg.attach(part)

    # 重试发送（最多 3 次）
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            print(f"  尝试发送 #{attempt}/{max_retries}...")
            if port == 465:
                server = smtplib.SMTP_SSL(host, port, timeout=30)
            else:
                server = smtplib.SMTP(host, port, timeout=30)
                server.starttls()
            server.login(user, passwd)
            recipients = [r.strip() for r in to.split(",")]
            # SMTP envelope from/to 必须是纯 ASCII（不能含中文显示名）
            server.sendmail(user, recipients, msg.as_string())
            server.quit()
            print(f"✅ 邮件发送成功！收件人: {to}")
            sys.exit(0)
        except smtplib.SMTPAuthenticationError as e:
            print(f"❌ 认证失败: {e}")
            print("   请检查 SMTP_USER 和 SMTP_PASS（QQ 邮箱用授权码不是密码）")
            sys.exit(1)
        except (smtplib.SMTPException, Exception) as e:
            print(f"❌ 发送失败(尝试{attempt}): {e}")
        if attempt < max_retries:
            wait = 10 * attempt
            print(f"   等待{wait}秒后重试...")
            time.sleep(wait)

    print(f"❌ {max_retries}次重试均失败")
    sys.exit(1)


if __name__ == "__main__":
    main()
