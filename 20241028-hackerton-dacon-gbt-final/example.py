# 패키지 import
import asyncio
import datetime # 비동기 라이브러리
from playwright.async_api import async_playwright, Playwright
from bs4 import BeautifulSoup

from src.database.session import SessionLocal
from src.database.models import EpeopleCaseOrm

WAIT_SHORT_TIME = 0.5
WAIT_NORMAL_TIME = 1

TARGET_PAGE = 3

async def run(playwright: Playwright) -> None:
    sess = SessionLocal()

    try:
        browser = await playwright.chromium.launch(headless=False) 
        context = await browser.new_context() 
        page = await context.new_page()
        await page.goto("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid")
        await asyncio.sleep(WAIT_SHORT_TIME)

        for page_num in range(1, TARGET_PAGE):
            for i in range(1, 11):
                # 1. Parse id of Target Case from the list page
                case_list_html = await page.content()
                html_parser = BeautifulSoup(case_list_html, 'html.parser')
                case_id = html_parser.select_one(f"#frm > table > tbody > tr:nth-child({i}) > td").text.strip()

                # 2. Go to the Target Case page and Parse the content(Question and Answer)
                await page.click(f"#frm > table > tbody > tr:nth-child({i}) > td.left > a")
                await asyncio.sleep(WAIT_SHORT_TIME) 
                html_content = await page.content()
                html_parser = BeautifulSoup(html_content, 'html.parser')

                # 2.1. Parse the content(Question)
                samBox_mw = html_parser.select_one("#txt > div.same_mwWrap > div.samBox.mw")
                title = samBox_mw.select_one("div.sam_cont > div.samC_top > strong").text.strip()
                question_date_str = samBox_mw.select_one("span.samC_date").text.strip()
                question_date = datetime.datetime.strptime(question_date_str, '%Y-%m-%d')

                # 2.2. Parse the content(Answer)
                samBox_ans = html_parser.select_one("#txt > div.same_mwWrap > div.samBox.ans")
                samC_top = samBox_ans.select_one("div.sam_cont > div.samC_top")
                content = samC_top.get_text(strip=True, separator='\n')
                answer_date_str = samC_top.select_one("span.samC_date").text.strip()
                answer_date = datetime.datetime.strptime(answer_date_str, '%Y-%m-%d')
                department = samBox_ans.select_one("div.samC_c ul.samC_info li:nth-child(1) dd").text.strip()
                related_laws = [law.text.strip() for law in samBox_ans.select("div.samC_c ul.samC_info li:nth-child(2) dd ul.samC_link li a")]

                # 3. Save the content to the database
                orm = EpeopleCaseOrm(
                    case_id=case_id,
                    title=title,
                    question_date=question_date, 
                    content=content,
                    department=department,
                    related_laws=','.join(related_laws),
                    answer_date=answer_date
                )
                sess.add(orm)
                sess.commit()
                
                # Back to the list
                await page.click("#txt > div.same_mwWrap > div.btnArea.right > button")
                await asyncio.sleep(WAIT_NORMAL_TIME)

            # Goto next page
            await page.click("#frm > div.page_list > span.nep_p_next")
            await asyncio.sleep(WAIT_NORMAL_TIME)
            page_num += 1
    except Exception as e: raise e
    finally:
        await context.close() # 콘텍스트 종료
        await browser.close() # 브라우저 종료
        sess.close()

async def main() -> None:
    async with async_playwright() as playwright:
        await run(playwright)

asyncio.run(main()) # 실행