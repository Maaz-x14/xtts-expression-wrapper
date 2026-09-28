
## English set (base model, port 8000, `--model base --language english`)

| #  | Emotion                        | Sentence                                                                                                                           |
| -- | ------------------------------ | ---------------------------------------------------------------------------------------------------------------------------------- |
| 1  | neutral                        | Normal sentence, longer — "The weather has been unusually cold this week, and everyone at the office keeps complaining about it." |
| 2  | happy                          | "I just got the internship offer and I honestly cannot stop smiling right now."                                                    |
| 3  | sad                            | "She left without saying goodbye, and the house has felt empty ever since."                                                        |
| 4  | angry                          | "You promised to call me back yesterday and you never even bothered."                                                              |
| 5  | fearful                        | "Something moved in the dark hallway and I froze, too scared to breathe."                                                          |
| 6  | disgust                        | "The food had gone bad days ago and the smell was absolutely revolting."                                                           |
| 7  | surprised                      | "Wait, you're telling me they got married last month without inviting anyone?"                                                     |
| 8  | calm                           | "Take a slow breath, relax your shoulders, and let the tension fade away."                                                         |
| 9  | happy (alias`joy`, prosodic) | "I finally finished the project <pause=400ms> and I could not be happier about it."                                                |
| 10 | angry (heavy prosodic combo)   | "This is<fast>completely unacceptable</fast> <pause=300ms> and I want an explanation <slow>right now</slow>."                      |

## Urdu set (finetuned model, port 8001, `--model finetuned --language urdu`)

| #  | Emotion                        | Sentence                                                                                                                             |
| -- | ------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------ |
| 1  | neutral                        | "آج موسم کافی خوشگوار ہے اور بازار میں کافی رونق نظر آ رہی ہے۔"                      |
| 2  | happy                          | "مجھے یقین نہیں آ رہا کہ ہم آخرکار یہ مقابلہ جیت گئے ہیں۔"                               |
| 3  | sad                            | "وہ بغیر بتائے چلا گیا اور اب گھر بالکل سنسان لگتا ہے۔"                                    |
| 4  | angry                          | "میں نے تم سے کل واپس فون کرنے کا کہا تھا اور تم نے نہیں کیا۔"                          |
| 5  | fearful                        | "اندھیرے کمرے میں اچانک آواز آئی اور میں خوف سے جم گیا۔"                                  |
| 6  | disgust                        | "کھانا کئی دن پرانا تھا اور اس کی بو بہت خراب تھی۔"                                            |
| 7  | surprised                      | "کیا واقعی انہوں نے بغیر بتائے شادی کر لی؟ مجھے یقین نہیں آ رہا۔"                  |
| 8  | calm                           | "آہستہ سے سانس لیں <pause=400ms> اور اپنے آپ کو پرسکون رکھیں۔"                                  |
| 9  | sad (alias`grief`, prosodic) | "اس کے جانے کے بعد <pause=500ms> گھر میں ایک عجیب سی خاموشی ہے۔"                                |
| 10 | angry (heavy prosodic combo)   | "یہ بالکل <pause=300ms> ناقابلِ قبول ہے<fast>اور مجھے ابھی</fast> <slow>جواب چاہیے۔</slow>" |
