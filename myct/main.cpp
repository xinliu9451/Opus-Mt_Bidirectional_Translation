#include <iostream>
#include <vector>
#include <src/sentencepiece_processor.h> // 包含 SentencePiece 支持
#include "ctranslate2/translator.h"

int main(int argc, char* argv[]) {
  if (argc < 3) {
    std::cerr << "Usage: " << argv[0] << " <source text>" << " <model name>"  << std::endl;
    return 1;
  }

  const std::string input_text = argv[1];
  const std::string model_name = argv[2];

  // 写死分词模型路径
  const std::string source_spm = "../" + model_name + "/source.spm";  // 替换为实际的 source spm 模型路径
  const std::string target_spm = "../" + model_name + "/target.spm";  // 替换为实际的 target spm 模型路径

  // 加载 SentencePiece 模型
  sentencepiece::SentencePieceProcessor source_processor;
  sentencepiece::SentencePieceProcessor target_processor;

  if (!source_processor.Load(source_spm).ok()) {
    std::cerr << "Failed to load source SentencePiece model: " << source_spm << std::endl;
    return 1;
  }
  if (!target_processor.Load(target_spm).ok()) {
    std::cerr << "Failed to load target SentencePiece model: " << target_spm << std::endl;
    return 1;
  }

  // 分词输入
  std::vector<std::string> tokens;
  source_processor.Encode(input_text, &tokens);

  // 打印分词结果（可选）
  std::cout << "Tokenized input: ";
  for (const auto& token : tokens) {
    std::cout << token << ' ';
  }
  std::cout << std::endl;

  // 添加结束符
  tokens.push_back("</s>");

  // 写死翻译器模型路径
  const std::string model_path("../" + model_name);  // 替换为实际的模型路径
  ctranslate2::Translator translator(model_path);

  // 翻译
  const std::vector<std::vector<std::string>> batch = {tokens};
  const auto translation = translator.translate_batch(batch);

  // 反分词输出
  std::vector<std::string> translated_tokens = translation[0].output();
  std::string detokenized_output;
  target_processor.Decode(translated_tokens, &detokenized_output);

  // 打印翻译结果
  std::cout << "Translated output: " << detokenized_output << std::endl;

  return 0;
}
