from argparse import ArgumentParser
from pathlib import Path
import time

from playwright.sync_api import sync_playwright


competition_id = {'ytvis19': '7682',
                  'ytvis21': '7680',
                  'ytvis22': '3410',
                  'ovis': '5857',
                  'vspw': '7869'}


def upload_file(result_dir, competition, account, password):
    num_submission = 0
    submission_list = []
    for _ in Path(result_dir).iterdir():
        submission_list.append(_)
        num_submission += 1

    submission_list.sort()
    with sync_playwright() as playwright:
        a_time = time.time()
        browser = playwright.chromium.launch()
        context = browser.new_context()
        # Open new page
        page = context.new_page()
        page.set_default_timeout(0)
        # Sign-in
        page.goto("https://codalab.lisn.upsaclay.fr/accounts/login/")
        page.locator("#id_login").fill(account)
        page.locator("#id_password").fill(password)
        page.locator("form.login button[type=submit]").click()
        b_time = time.time()
        print(f"Congrats! Login in.It takes {(time.time() - a_time):.1f}s.")
        # Upload zip
        for i, submission in enumerate(submission_list):
            if Path(submission).suffix == ".zip":
                page.goto(f"https://codalab.lisn.upsaclay.fr/competitions/{competition_id[competition]}#participate-submit_results")
                page.locator("#s3_upload_form input[type=file]").set_input_files(submission)
                page.locator("#s3-file-upload").click()
                print(f"[{i + 1} / {num_submission}] {submission} upload finished.")
            else:
                print(f"[{i + 1} / {num_submission}] {submission} is not vaild.")
        end_time = time.time()
        print(f"Finish {num_submission} submissions.It takes {(end_time - b_time):.1f}s. Average {((end_time - b_time) / num_submission):.1f}s per submission.")
        # Close page
        context.close()
        browser.close()


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--result-dir', default="_temp", help='GT file directory')
    parser.add_argument('--id', default="ytvis19", help='ID')
    parser.add_argument('--account', help='CodaLab account')
    parser.add_argument('--password', help='CodaLab password')

    args = parser.parse_args()
    return args


def main(args):
    result_dir = args.result_dir
    upload_file(result_dir, args.id, args.account, args.password)


if __name__ == '__main__':
    args = parse_args()
    main(args)
