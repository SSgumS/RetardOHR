initials_org = 'ئ ب پ ت ث ج چ ح خ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن ه ی'
initials = 'ﺋ ﺑ ﭘ ﺗ ﺛ ﺟ ﭼ ﺣ ﺧ ﺳ ﺷ ﺻ ﺿ ﻃ ﻇ ﻋ ﻏ ﻓ ﻗ ﮐ ﮔ ﻟ ﻣ ﻧ ﻫ ﯾ'
medials_org = 'ئ ب پ ت ث ج چ ح خ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن ه ی'
medials = 'ﺌ ﺒ ﭙ ﺘ ﺜ ﺠ ﭽ ﺤ ﺨ ﺴ ﺸ ﺼ ﻀ ﻄ ﻈ ﻌ ﻐ ﻔ ﻘ ﮑ ﮕ ﻠ ﻤ ﻨ ﻬ ﯿ'
finals_org = 'أ ؤ ئ آ ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی إ ة'
finals = 'ﺄ ﺆ ﺊ ﺂ ﺎ ﺐ ﭗ ﺖ ﺚ ﺞ ﭻ ﺢ ﺦ ﺪ ﺬ ﺮ ﺰ ﮋ ﺲ ﺶ ﺺ ﺾ ﻂ ﻆ ﻊ ﻎ ﻒ ﻖ ﮏ ﮓ ﻞ ﻢ ﻦ ﻮ ﻪ ﯽ ﺈ ﺔ'
isolateds = 'ء أ آ ا ب پ ت ث ج چ ح خ د ذ ر ز ژ س ش ص ض ط ظ ع غ ف ق ک گ ل م ن و ه ی ؤ ئ إ ة'
void_glyphs = 'َ ِ ُ ّ ً ٍ ٌ ْ'

initials_org = initials_org.split(' ')
initials = initials.split(' ')
medials_org = medials_org.split(' ')
medials = medials.split(' ')
finals_org = finals_org.split(' ')
finals = finals.split(' ')
isolateds = isolateds.split(' ')
void_glyphs = void_glyphs.split(' ')
triggers = set(initials_org + medials_org + finals_org)


def normalize(text: str):
    text_size = len(text)
    new_text = ''
    for i in range(text_size):
        if i > 0:
            last_i = i - 1
            last_letter = new_text[last_i]
            while last_letter in void_glyphs:
                last_i -= 1
                if last_i == -1:
                    last_letter = None
                    break
                last_letter = new_text[last_i]
        else:
            last_letter = None

        letter = text[i]
        if letter not in triggers:
            new_text += letter
            continue

        if i < text_size - 1:
            next_i = i + 1
            next_letter = text[next_i]
            while next_letter in void_glyphs:
                next_i += 1
                if next_i == text_size:
                    next_letter = None
                    break
                next_letter = text[next_i]
        else:
            next_letter = None

        added = False

        # initial
        if (last_letter not in initials and last_letter not in medials) \
                and (next_letter in medials_org or next_letter in finals_org) \
                and letter in initials_org:
            new_text += initials[initials_org.index(letter)]
            added = True

        # medial
        if (last_letter in initials or last_letter in medials) \
                and (next_letter in medials_org or next_letter in finals_org) \
                and letter in medials_org:
            new_text += medials[medials_org.index(letter)]
            added = True

        # final
        if (last_letter in initials or last_letter in medials) \
                and ((letter not in initials_org and letter not in medials_org)
                     or (next_letter not in medials_org and next_letter not in finals_org)) \
                and letter in finals_org:
            new_text += finals[finals_org.index(letter)]
            added = True

        if not added:
            new_text += letter
    return new_text


def unnormalize(normalized_text: str):
    text_size = len(normalized_text)
    new_text = ''
    for i in range(text_size):
        letter = normalized_text[i]

        if letter in initials:
            new_text += initials_org[initials.index(letter)]
        elif letter in medials:
            new_text += medials_org[medials.index(letter)]
        elif letter in finals:
            new_text += finals_org[finals.index(letter)]
        else:
            new_text += letter
    return new_text


def normalize_and_check(text: str):
    normalized_text = normalize(text)
    original_text = unnormalize(normalized_text)
    assert text == original_text
    return normalized_text
