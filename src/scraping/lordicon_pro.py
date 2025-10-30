from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import time
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver import ActionChains
import os
from dotenv import load_dotenv
from selenium.common.exceptions import TimeoutException
import pickle
def js_click(el):
    # Scroll to center and click with a real MouseEvent that bubbles across shadows
    driver.execute_script("""
      const el = arguments[0];
      el.scrollIntoView({block: 'center', inline: 'nearest'});
      const ev = new MouseEvent('click', {bubbles: true, cancelable: true, composed: true});
      el.dispatchEvent(ev);
    """, el)

login = (1529, 35)
export = (1522, 865)

download = (967, 572)
lottie = (733, 341)
cookies = (1324, 778)
cross = (1345, 303)
options = (1773, 641)
coll = (1849, 103)
fav = (895, 414)
fav2 = (932,371)
fav_clicked = False
crossed = False
clicked_free = True
lottie_clicked = False


def click(x, y):
    # Example: click at (x=200, y=300) relative to the page
    
    actions = ActionChains(driver)
    actions.move_by_offset(x, y).click().perform()

    # Reset the mouse back to (0,0) to avoid offset issues for later clicks
    actions.move_by_offset(-x, -y).perform()

def qs_js(root, selector):
    # querySelector on either document or a shadowRoot
    return driver.execute_script("return arguments[0].querySelector(arguments[1]);", root, selector)

def get_shadow_root(host):
    return driver.execute_script("return arguments[0].shadowRoot;", host)

def find_in_nested_shadows(selector_chain, final_selector):
    """
    selector_chain: list of shadow hosts to descend through (outer → inner)
    final_selector: CSS selector to return from the deepest shadowRoot
    """
    root = driver.execute_script("return document;")  # start at document
    for sel in selector_chain:
        host = wait.until(lambda d: qs_js(root, sel))
        root = get_shadow_root(host)
        if root is None:
            raise RuntimeError(f"Host '{sel}' has no shadowRoot")
    return wait.until(lambda d: qs_js(root, final_selector))

# def click_free():

#     shadow_host = driver.find_element(By.CSS_SELECTOR, "li-library-sidebar")

#     # get its shadow root
#     shadow_root = driver.execute_script("return arguments[0].shadowRoot", shadow_host)

#     # now you can query inside the shadow root
#     items = shadow_root.find_elements(By.CSS_SELECTOR, "div.header")

#     for item in items:
#         if "Filters" in item.get_attribute("textContent"):
#             item.click()

#     driver.execute_script("window.scrollBy(0, 100);")

#     time.sleep(2)
#     export_div = find_in_nested_shadows(
#         ["li-library-sidebar",],
#         "li-field-checkbox"
#     )
#     export_div.click()
    # js_click(export_div)
    
opts = Options()
opts.add_argument("--start-maximized")
# opts.add_argument("--headless=new")  # if needed
driver = webdriver.Chrome(options=opts)  # Selenium Manager fetches the right driver


driver.get("https://lordicon.com/icons/system/solid?f=free")
time.sleep(2)

click(*cookies)
time.sleep(0.5)
# login now ###########################################
click(*login)

load_dotenv()
EMAIL = os.getenv("EMAIL")
PW = os.getenv("PW")

wait = WebDriverWait(driver, 10)
# Fill email
email_input = wait.until(EC.presence_of_element_located((By.NAME, "email")))
email_input.clear()
email_input.send_keys(EMAIL)

# Fill password
password_input = driver.find_element(By.NAME, "password")
password_input.clear()
password_input.send_keys(PW)

# Click submit (find by class)
submit_btn = driver.find_element(By.CSS_SELECTOR, "div.field-submit button")
submit_btn.click()
time.sleep(2)
######################################################

def wait_overlay_closed(timeout=10):
    # returns when the inner menu element is NOT present in the shadow DOM
    WebDriverWait(driver, timeout).until(
        lambda d: d.execute_script("""
            const host = document.querySelector('li-overlay-outlet');
            if (!host || !host.shadowRoot) return true;
            // this element exists only while the overlay/menu is open
            return !host.shadowRoot.querySelector('li-field-editor-state-list');
        """)
    )

thingys  = [
    # "https://lordicon.com/icons/wired/outline",
    # "https://lordicon.com/icons/system/regular",
    "https://lordicon.com/icons/system/solid"
    # "https://lordicon.com/icons/wired/flat",
    # "https://lordicon.com/icons/wired/lineal",
    # "https://lordicon.com/icons/wired/gradient",
]
uniques_path = "unique.pkl"

if os.path.exists(uniques_path):
    with open(uniques_path, 'rb') as f:
        unique = pickle.load(f)
else:
    unique = set()

# print(unique)
print(f"{len(unique)} found.")
for url in thingys:
    driver.get(url)
    time.sleep(2)

    # if not clicked_free:
    #     click_free()
    #     time.sleep(1)
    # find the shadow host element
    shadow_host = driver.find_element(By.CSS_SELECTOR, "li-library-sidebar")
    # get its shadow root
    shadow_root = driver.execute_script("return arguments[0].shadowRoot", shadow_host)

    # now you can query inside the shadow root
    items = shadow_root.find_elements(By.CSS_SELECTOR, "li.category")
    n_items = len(items)
    wait = WebDriverWait(driver, 15)

    # input()
    # unique = set()

    # Get all categories
    # for item in items:
    for i in range(1, n_items):
        driver.refresh()
        time.sleep(1)


        shadow_host = driver.find_element(By.CSS_SELECTOR, "li-library-sidebar")

        # get its shadow root
        shadow_root = driver.execute_script("return arguments[0].shadowRoot", shadow_host)

        # now you can query inside the shadow root
        items = shadow_root.find_elements(By.CSS_SELECTOR, "li.category")

        item = items[i]
        # Click category
        # item.click()
        js_click(item)
        time.sleep(1)

        god_shadow_host = WebDriverWait(driver, 5).until(
            EC.presence_of_element_located((By.CSS_SELECTOR, "li-library-icons"))
        )

        # 2. Get its shadow root
        god_shadow_root = driver.execute_script("return arguments[0].shadowRoot", god_shadow_host)

        # 3. Find all elements with class 'icon' inside that shadow root
        icons = god_shadow_root.find_elements(By.CSS_SELECTOR, ".icon")
        n_icons = len(icons)

        # for all icons, download the,
        # for i, icon in enumerate(icons):
        for i in range(n_icons):
            
            driver.refresh()
            time.sleep(1)
            god_shadow_host = WebDriverWait(driver, 5).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, "li-library-icons"))
            )

            # 2. Get its shadow root
            god_shadow_root = driver.execute_script("return arguments[0].shadowRoot", god_shadow_host)

            # 3. Find all elements with class 'icon' inside that shadow root
            icons = god_shadow_root.find_elements(By.CSS_SELECTOR, ".icon")
            icon = icons[i]

            icon_in = icon.find_element(By.CSS_SELECTOR, "lord-icon")
            src = icon_in.get_attribute("src")
            print(src.split("/")[-1])

            if src not in unique:
                unique.add(src)
            else:
                continue
            driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", icon)
            print(f"Icon {i}:")
            time.sleep(0.5)
            
            # icon.click()
            try:
                icon.click()
            except Exception:
                print("JS CLICK")
                js_click(icon)

            # time.sleep(1)
            time.sleep(0.5)


            # Click options
            # click(*options)

            host1 = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li-library-editor")))
            option = host1.shadow_root.find_element(By.CSS_SELECTOR, "li-field-editor-state-select")
            option.click()
            time.sleep(0.5)

            
            host1 = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li-overlay-outlet")))
            root1 = host1.shadow_root

            # 2) <li-field-editor-state-list>  →  #shadow-root (open)
            host2 = root1.find_element(By.CSS_SELECTOR, "li-field-editor-state-list")
            root2 = host2.shadow_root

            # 3) divs with class="group" inside the inner shadow root
            groups = root2.find_elements(By.CSS_SELECTOR, "span")
            # print(groups)
            # input()
            for grp in groups:
                driver.execute_script("arguments[0].scrollIntoView({block: 'center', inline: 'center'});", grp)

                print(grp.text)
                time.sleep(1)

                # grp.click()
                js_click(grp)
                time.sleep(0.1)

                click(*export)
                time.sleep(0.5)

                if not lottie_clicked:
                    click(*lottie)
                    time.sleep(1)
                    lottie_clicked = True
                click(*download)
                time.sleep(0.5)

                host1 = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li-library-editor")))
                option = host1.shadow_root.find_element(By.CSS_SELECTOR, "li-field-editor-state-select")
                option.click()
                time.sleep(0.5)

            with open(uniques_path, 'wb') as f:
                pickle.dump(unique, f)
                    # input()
            # continue

            # Add to coll
            # click(*coll)
            # time.sleep(1)

            
            # click(*fav)
            # time.sleep(1)
            # if not fav_clicked:
            #     fav = fav2
            #     fav_clicked = True

            # # Click option
            # click(*options)
            # time.sleep(0.5)

            # host1 = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, "li-overlay-outlet")))
            # root1 = host1.shadow_root

            # # 2) <li-field-editor-state-list>  →  #shadow-root (open)
            # host2 = root1.find_element(By.CSS_SELECTOR, "li-field-editor-state-list")
            # root2 = host2.shadow_root

            # # 3) divs with class="group" inside the inner shadow root
            # groups = root2.find_elements(By.CSS_SELECTOR, "span")
            # # for all option:
            # # add to coll
            # for g in groups:
            #     click(*coll)
            #     time.sleep(1)

            #     click(*fav)
            #     time.sleep(1)

            #     click(*options)
            #     time.sleep(0.5)

                # print(g.text)
                # g.click()
                # time.sleep(0.5)
            
                # click(*export)
                # time.sleep(0.5)
                # click(*lottie)
                # time.sleep(1)
                # click(*download)

                # click(*options)
                # time.sleep(0.5)

                # if not crossed:
                #     time.sleep(3)
                #     click(*cross)
                #     crossed=True

            # click(*export)
            # time.sleep(0.5)
            # click(*lottie)
            # time.sleep(1)
            # click(*download)

            # if not crossed:
            #     time.sleep(3)
            #     click(*cross)
            #     crossed=True

            
            # input()

            

            


           

            # print(f"found {len(groups)} groups")
          
            
            # input("")