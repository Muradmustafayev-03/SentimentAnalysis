MAX_WORDS = 3000
MAX_TEXT_LEN = 30
OUTPUT_SIZE = 14  # number of distinct emotions in the datasets
FILTERS = '–"—#$%&amp;()*+,-./:;<=>?@[\\]^_`{|}~\t\n\r«»'
EXAMPLES = {
    'The sun is shining, the birds are singing, and life is full of endless possibilities!': 'happy',
    'The once-bustling streets are now silent and empty, as the world grapples with the devastating effects of a global pandemic.': 'sad',
    'I am infuriated by the blatant disregard for basic human rights and equality that continues to plague our society.': 'angry',
    'The sound of thunder in the distance filled her with fear, as she knew a storm was brewing and she was alone in the house.': 'fearful',
    'As she waited for the exam results to be posted online, her heart raced with anxiety and she couldn\'t stop second-guessing her answers.': 'anxious',
    'I can hardly contain my excitement about the upcoming trip to Europe - the thought of exploring new cities and immersing myself in different cultures fills me with anticipation!': 'excited',
    'Staring at the clock, watching the minutes tick by, he couldn\'t help but feel bored out of his mind in the never-ending meeting.': 'bored',
    'As she looked around the dinner table surrounded by her loved ones, she felt an overwhelming sense of gratitude for the warmth and comfort of their company.': 'grateful',
    'He couldn\'t shake the feeling of guilt as he realized he had forgotten his best friend\'s birthday, despite promising to make it special.': 'guilty',
    'The way that people from all over the world came together to support each other during the pandemic gives me hope that we can overcome any challenge when we stand united.': 'hopeful',
    'Whenever I\'m with you, my heart skips a beat and I can\'t help but feel an overwhelming sense of love and happiness.': 'love',
    'She felt an immense sense of pride as she watched her daughter receive her diploma, knowing that all her hard work and dedication had paid off in such a meaningful way.': 'proud',
    'He looked back on his life with regret, wishing he had taken more risks and pursued his passions instead of always playing it safe.': 'regretful',
    'The cat sat lazily on the windowsill, watching the world go by outside.': 'neutral'
}
EMOTIONS = sorted(['regretful', 'angry', 'guilty', 'excited', 'proud', 'hopeful', 'bored',
                   'neutral', 'grateful', 'fearful', 'love', 'anxious', 'sad', 'happy'])
