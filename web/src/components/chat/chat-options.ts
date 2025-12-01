export interface ChatOptions {
  selectedModel: string;
  systemPrompt: string;
  temperature: number;
  topP: number;
  topK: number;
  minP: number;
  includeThinking: boolean;
}