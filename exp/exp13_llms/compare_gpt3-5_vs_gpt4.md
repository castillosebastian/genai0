# evaluations gpt.3-5 vs gpt.4

- MemGPT, 5.1. Limitations: 'While GPT models that have been finetuned for function-calling still require a parser to verify outputs as valid function syntax, we observed that GPT-4 function fine-tuned models rarely made syntactic or semantic errors on the MemGPT function set, whereas GPT-3.5 finetuned models consistently generated incorrect function calls, or used attempted to use functions incorrectly.' arXiv:2310.08560.
  
- FinnanceBench: 'During our early testing, we compared GPT-4 against GPT-3.5. GPT3.5’s performance was similar but slightly worse. Due to this, we decided not to continue testing GPT-3.5 further.' arXiv:2311.11944, p.5

- Context-Windows: GPT-3.5-turbo supports a context window of 4096 tokens, while gpt-4 supports 8192. FinanceBench used entirely pages as context/chunk.  One page context size reference:
  1) 1 page 3M-10k-2022, BalanceSheet = 285 tokens / 2000 characters 
  2) 1 page 3M-10k-2022, MD&A = 707 tokens / 4650 characters 

- Statistics (arXiv:2310.08560)
  1) gpt-3.5-turbo API 4k tokens           60 total messages     
  2) gpt-4 API 8k tokens                   140 total messages     
  3) gpt-3.5-turbo-16k API 16k tokens      300 total messages      


