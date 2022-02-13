from rouge import Rouge

hypothesis = ["the #### transcript is a written version of each day 's cnn student news program use this transcript to help students with reading comprehension and vocabulary use the weekly newsquiz to test your knowledge of storie s yousaw on cnn student news" for i in range(2)]

reference = ["this page includes the show transcript use the transcript to help students with reading comprehension and vocabulary at the bottom of the page , comment for a chance to be mentioned on cnn student news . you must be a teacher or a student age # # or older to request a mention on the cnn student news roll call . the weekly newsquiz tests     students ' knowledge of even ts in the news" for i in range(2)]

print("=="*10)
print("hypothesis", " ".join(hypothesis))

print("=="*10)
print("reference", ". ".join(reference))

rouge = Rouge()
scores = rouge.get_scores(hypothesis, reference, avg=True)

print("scores", scores)

print(scores["rouge-1"]["f"])