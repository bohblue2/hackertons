# 패키지 import
import asyncio
import datetime
from sqlite3 import IntegrityError # 비동기 라이브러리
from playwright.async_api import async_playwright, Playwright
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from yongin_assistant.database.session import SessionLocal
from yongin_assistant.database.models import EpeopleCaseOrm

WAIT_SHORT_TIME = 1
WAIT_NORMAL_TIME = 2
WAIT_LONG_TIME = 4
USE_PAGE_CORDINATION = True
START_PAGE = 121
TARGET_PAGE = 1000
ERROR_COUNT_THRESHOLD = 10
RETRY_EXCEPTIONS = (TimeoutError, ConnectionError)  # 필요한 예외 추가

@retry(
    retry=retry_if_exception_type(RETRY_EXCEPTIONS),
    stop=stop_after_attempt(ERROR_COUNT_THRESHOLD),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    reraise=True
)
async def parse_case(page, page_num:int, idx:int, total_count:int, sess: SessionLocal) -> None: # type: ignore
    # 1. Parse id of Target Case from the list page
    case_list_html = await page.content()
    html_parser = BeautifulSoup(case_list_html, 'html.parser')
    case_id = html_parser.select_one(f"#frm > table > tbody > tr:nth-child({idx}) > td").text.strip()
    print(f"[{total_count}] PAGE {page_num+START_PAGE - 1}- IDX {idx} - {case_id}")

    # 2. Go to the Target Case page and Parse the content(Question and Answer)
    await page.click(f"#frm > table > tbody > tr:nth-child({idx}) > td.left > a")
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
    try:
        sess.commit()
    except IntegrityError as e:
        # sqlalchemy.exc.IntegrityError: (sqlite3.IntegrityError) UNIQUE constraint failed: epeople_cases.case_id
        sess.rollback()
        print(f"Already Exists: {case_id}")
    except Exception as e:
        raise e

    # Back to the list
    await page.click("#txt > div.same_mwWrap > div.btnArea.right > button")
    await asyncio.sleep(WAIT_NORMAL_TIME)


async def run(playwright: Playwright) -> None:
    sess = SessionLocal()
    total_count = 0
    error_count = 0

    try:
        browser = await playwright.chromium.launch(headless=False) 
        context = await browser.new_context() 
        page = await context.new_page()
        await page.goto("https://www.epeople.go.kr/nep/pttn/gnrlPttn/pttnSmlrCaseList.npaid")
        await asyncio.sleep(WAIT_SHORT_TIME)

        if START_PAGE != 1:
            await page.evaluate(f"frmPageLink({START_PAGE})")
            await page.wait_for_load_state('networkidle')
            await asyncio.sleep(WAIT_LONG_TIME)
        if USE_PAGE_CORDINATION:
            start_idx = int(input("Press Enter to continue...")) # 9
        for page_num in range(1, TARGET_PAGE):
            for i in range(start_idx, 11):
                try:    
                    await parse_case(page, page_num, i, total_count, sess)
                except Exception as e:
                    error_count += 1
                    print(f"Error Count: {error_count}")
                    if error_count > ERROR_COUNT_THRESHOLD:
                        raise e
                    continue
                else:
                    total_count += 1

            # Goto next page
            await page.click("#frm > div.page_list > span.nep_p_next")
            await asyncio.sleep(WAIT_NORMAL_TIME)
            page_num += 1
            
            # After moved to the next page, reset the start_idx 
            if start_idx != 1:
                start_idx = 1

    except Exception as e: 
        
        raise e

    finally:
        await context.close() # 콘텍스트 종료
        await browser.close() # 브라우저 종료
        sess.close()

async def main() -> None:
    async with async_playwright() as playwright:
        await run(playwright)

asyncio.run(main()) # 실행