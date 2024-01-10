import requests
response=requests.get("https://api.github.com/repos/SkafteNicki/dtu_mlops")
content = response.json()
#print(content)
print(response.status_code)
if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')