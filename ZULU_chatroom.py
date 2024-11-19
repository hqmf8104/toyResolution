from toyMaven.chatFunctions import chatToEntites
textDict = chatToEntites()
[print(f"{ii}: {textDict[ii]}") for ii in textDict.keys()]
#[id]:[(x,y),TOI, description,identification(friend, enemy, unk), [file_loc]]


# Schedule the function to run periodically (every 10 seconds, for example)
# import schedule
#schedule.every(10).seconds.do(collect_new_messages)

#while True:
#    schedule.run_pending()
#    time.sleep(1)



"""
1. Get chat history
2. Generate a list of unprocessed messages
3. Iterate over those messages and process
    a. Is it a track? 
    b. If yes, ensure it's in the correct format.
"""
