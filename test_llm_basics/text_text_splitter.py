import os
from langchain.text_splitter import NLTKTextSplitter
from langchain.docstore.document import Document

p = """
Sales in the first three quarters of 2020 totaled € 453.9 million. Following € 467.3
million in the first nine months of 2019, this represents a decrease by € 13.5 million,
or 2.9 %, respectively. Based on a somewhat weaker overall demand situation sales
in the third quarter of the current fiscal year totaled € 152.0 million and were thus
slightly below the level of Q3/2019 (€ 156.2 million). The strongest sales development
in the first three quarters 2020 was recorded in the market segment semiconductors,
which was able to almost fully compensate the downtrend in the other segments. In
total, the gross profit decreased by € 3.4 million, or 2.1 %, respectively, to € 158.0
million (2019: € 161.4 million). Particularly changes in product mix improved the gross
margin slightly from 34.5% to 34.8 %. Due to the Group’s focus on further growth
general and administrative expenses recorded an increase by € 2.5 million to € 43.5
compared to the first three quarters of 2019 (€ 40.9 million). The R & D expenses
increased by € 4.3 million to € 25.5 million for the first three quarters of 2020
compared to € 21.2 million for the first three quarters of 2019. Selling and marketing
expenses on the contrary only showed slight increases in the first nine months of
2020 compared to the previous year. The balance of other operating income and
expenses declined by € 4.8 million to € 0.6 million compared to previous year. In total,
an operating profit of € 35.7 million was generated in the first three quarters of the
current fiscal year, down by € 13.2 million, or 27.5 %, compared to previous year’s
value of € 48.9 million. As a consequence the EBIT margin, the ratio between
operating profit and sales, decreased from 10.5 % in 2019 to 7.9 %. With virtually
constant net financial expenses and a slightly increased tax rate, net income
decreased from € 34.7 million to € 24.8 million. This led to earnings per share of
€ 2.51 (2019: € 3.52).
    As mentioned before sales in the market semiconductors showed a positive trend and
therefore influenced the development in the category semiconductors and coating. In
contrast the demand in the coating market stayed quite challenging. Due to the
COVID-19 situation the sales with customers from analytics, industry and R&D
declined compared to previous year
"""

# docs = Document(page_content=p)
# splitter = NLTKTextSplitter(chunk_size=300, chunk_overlap=150)
# for text in splitter.split_text(p):
#     print("NEW--> ", text)


x = ['/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-4-1.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-1-1.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-2-1.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-2-2.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-1.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-2.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-4.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-5.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-7.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-2-3.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-3.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-6.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-8.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-3-11.jpg', '/home/jtg-d305/Desktop/openAI/intelligencia-assistant/test_llm_basics/figures/figure-4-2.jpg']\

print(len(x))
for xx in x:
    print(os.path.basename(xx))
