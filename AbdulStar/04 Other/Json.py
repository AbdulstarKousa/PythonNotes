"""
jason.load():
    * jason formated string -> python object
"""
import json
json_string = """
                {
                    "people": [
                        {
                            "name": "John Smith",
                            "age" : 63,
                            "isProgramer": false 
                        },
                        {
                            "name": "Abdulstar Kousa",
                            "age" : 29,
                            "isProgramer": true 
                        }
                    ]
                }
            """
data= json.loads(json_string)
print(data['people'][1]['name'])



"""
jason.dumps():
    * Python dic -> jason formated string 
"""
for person in data['people']:
    del person['age'] 
new_jason_string =  json.dumps(data,indent=2)
print(new_jason_string)



"""
Notes:
    * You could use indent=2, 
    * to make it easy to read when printing the new string,
    * see below:
        json.dumps(data,indent=2)
"""