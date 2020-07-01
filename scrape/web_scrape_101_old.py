# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 10:05:07 2020

@author: Grey Ghost
"""
from selenium import webdriver
import requests
import time
import pandas as pd
import inspect


url = 'https://www.xcontest.org/canada/en/flights/'
r = requests.get(url)

'''
url2 ='https://www.xcontest.org/world/en/flights/#filter[country]=CA@flights[sort]=reg'
r2 = requests.get(url2)
print(r2.text)
'''

driver = webdriver.Chrome()
#driver2 = webdriver.PhantomJS(executable_path = 'c:\apps\webdrivers\phantomjs.exe')
# get web page
driver.get(url)
# execute script to scroll down the page
time.sleep(20)
'''
//*[@id="flights"]/table
/html/body/div/div[1]/div[3]/div/div[1]/div[1]/div[3]/div[3]/table
'''
results = driver.find_elements_by_xpath('/html/body/div/div[1]/div[3]/div/div[1]/div[1]/div[3]/div[3]/table/tbody/tr/td[11]/div/a')
urls = []
for r in results:
    urls.append(r.get_attribute('href'))
df1 = pd.DataFrame(urls)
df1.to_csv('./flights.csv')
print('got the flights', len(urls), "in total")

count = 0
tests = urls[0:2]
for line in tests:
        print("-----------------\n",line)
        driver.get(line)
        time.sleep(5)
        file = driver.find_element_by_xpath('/html/body/div/div[1]/div[3]/div/div[1]/div[1]/div[3]/div[2]/div[1]/div[1]/table/tbody/tr[7]/th[1]/a')
        file2 = driver.find_element_by_xpath("/html/body/div/div[1]/div[3]/div/div[1]/div[1]/div[3]/div[2]/div[1]/div[1]/table/tbody/tr[7]/th[1]")
        file3 = driver.find_element_by_xpath("/html/body/div/div[1]/div[3]/div/div[1]/div[1]/div[3]/div[2]/div[1]/div[1]/table/tbody/tr[7]/th[1]/a")
        print(type(file), type(file2), type(file3))
        #print(len(file), len(file2), len(file3))
        try:
            file[0].click()
            print("yes file[0]")
        except:
            time.sleep(1)
            print("no file[0]")
        try:
            file2[0].click()
            print("yes file2[0]")
        except:
            time.sleep(1)
            print("no file2[0]")
        try:
            file3[0].click()
            print("yes file3[0]")
        except:
            time.sleep(1)
            print("no file3[0]")
        try:
            file.click()
            print("yes file")
        except:
            time.sleep(1)
            print("no file")
        try:
            file2.click()
            print("yes file2")
        except:
            time.sleep(1)
            print("no file2")
        try:
            file3.click()
            print("yes file3")
        except:
            time.sleep(1)
            print("no file3")
        time.sleep(5)
inspect.getmembers(file)
inspect.getmembers(file2)                 
inspect.getmembers(file3)          
        
driver.quit()

print('done!')

