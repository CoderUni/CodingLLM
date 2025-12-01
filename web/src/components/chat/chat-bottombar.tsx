"use client";

import React from "react";
import { StopIcon } from "@radix-ui/react-icons";
import { ChatRequestOptions } from "ai";
import llama3Tokenizer from "llama3-tokenizer-js";
import TextareaAutosize from "react-textarea-autosize";
import { Sparkles } from "lucide-react"; 

import { basePath, useHasMounted } from "@/lib/utils";
import { getTokenLimit } from "@/lib/token-counter";
import { Button } from "../ui/button";
import { cn } from "@/lib/utils";
import { ChatOptions } from "./chat-options";

interface ChatBottombarProps {
  selectedModel: string | undefined;
  input: string;
  handleInputChange: (e: React.ChangeEvent<HTMLTextAreaElement>) => void;
  handleSubmit: (
    e: React.FormEvent<HTMLFormElement>,
    chatRequestOptions?: ChatRequestOptions
  ) => void;
  isLoading: boolean;
  stop: () => void;
  chatOptions: ChatOptions;
  setChatOptions: React.Dispatch<React.SetStateAction<ChatOptions>>;
}

export default function ChatBottombar({
  selectedModel,
  input,
  handleInputChange,
  handleSubmit,
  isLoading,
  stop,
  chatOptions,
  setChatOptions,
}: ChatBottombarProps) {
  const hasMounted = useHasMounted();
  const inputRef = React.useRef<HTMLTextAreaElement>(null);
  const hasSelectedModel = selectedModel && selectedModel !== "";

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey && hasSelectedModel && !isLoading) {
      e.preventDefault();
      handleSubmit(e as unknown as React.FormEvent<HTMLFormElement>);
    }
  };

  const [tokenLimit, setTokenLimit] = React.useState<number>(4096);
  React.useEffect(() => {
    getTokenLimit(basePath).then((limit) => setTokenLimit(limit));
  }, [hasMounted]);

  const tokenCount = React.useMemo(
    () => (input ? llama3Tokenizer.encode(input).length - 1 : 0),
    [input]
  );

  const isThinkingEnabled = chatOptions?.includeThinking !== false;

  const toggleThinking = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    
    setChatOptions((prev) => {
      const currentVal = prev.includeThinking !== false; 
      return {
        ...prev,
        includeThinking: !currentVal,
      };
    });
  };

  return (
    <div className="w-full py-4 bg-background z-50">
      <div className="w-full max-w-3xl mx-auto px-4 flex flex-col gap-2">
        
        <form
            onSubmit={handleSubmit}
            className={cn(
                "w-full relative bg-muted/30 border border-input rounded-xl shadow-sm transition-all duration-200 flex flex-col",
                "focus-within:border-blue-500 focus-within:ring-1 focus-within:ring-blue-500"
            )}
        >
            <TextareaAutosize
              autoComplete="off"
              value={input}
              ref={inputRef}
              onKeyDown={handleKeyPress}
              onChange={handleInputChange}
              name="message"
              placeholder="Ask vLLM anything..."
              minRows={1}
              maxRows={8}
              className={cn(
                "w-full resize-none bg-transparent px-4 py-3 text-[15px] focus:outline-none placeholder:text-muted-foreground/60 overflow-y-auto scrollbar-thin scrollbar-thumb-gray-200 dark:scrollbar-thumb-gray-700",
                "min-h-[40px]" 
              )}
            />

            <div className="flex justify-between items-center p-2">
                
                <div className="z-20"> 
                   <Button
                      type="button"
                      onClick={toggleThinking}
                      variant="ghost"
                      size="sm"
                      className={cn(
                        "h-8 px-3 gap-2 rounded-full text-xs font-medium transition-all border",
                        isThinkingEnabled 
                          ? "!bg-blue-100 !text-blue-600 !border-blue-200 hover:!bg-blue-200 dark:!bg-blue-900/40 dark:!text-blue-300 dark:!border-blue-800 dark:hover:!bg-blue-900/60" 
                          : "bg-transparent text-muted-foreground border-transparent hover:bg-muted hover:text-foreground"
                      )}
                   >
                      <Sparkles className={cn("w-3.5 h-3.5", isThinkingEnabled ? "fill-current" : "")} />
                      <span>Thinking</span>
                   </Button>
                </div>

                <div className="flex items-center gap-3">
                    {input.length > 0 && (
                        <div className="text-xs text-muted-foreground/50 pointer-events-none select-none hidden sm:block">
                            {tokenCount > tokenLimit ? (
                            <span className="text-red-500 font-medium">
                                {tokenCount} / {tokenLimit}
                            </span>
                            ) : (
                            <span>{tokenCount} / {tokenLimit}</span>
                            )}
                        </div>
                    )}

                    {!isLoading ? (
                        <Button
                        size="icon"
                        className={cn(
                            "h-8 w-8 rounded-lg transition-all duration-200 shrink-0",
                            "!border-0",
                            // Active State (Blue)
                            "!bg-blue-600 !text-white hover:!bg-blue-700 hover:!shadow-sm hover:!scale-105 active:!scale-100",
                            // Disabled State (Transparent Background + Gray Icon)
                            "disabled:!bg-transparent disabled:!text-muted-foreground disabled:!opacity-100 disabled:cursor-not-allowed disabled:!shadow-none"
                        )}
                        type="submit"
                        disabled={!input.trim() || !hasSelectedModel}
                        >
                        <svg 
                            xmlns="http://www.w3.org/2000/svg" 
                            viewBox="0 0 24 24" 
                            fill="currentColor" 
                            className="w-4 h-4 ml-0.5" 
                        >
                            <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" />
                        </svg>
                        </Button>
                    ) : (
                        <Button
                        size="icon"
                        className={cn(
                            "h-8 w-8 rounded-lg shrink-0 !border-0",
                            "!bg-blue-600 !text-white hover:!bg-blue-700 hover:!shadow-sm hover:!scale-105 active:!scale-100"
                        )}
                        onClick={(e) => {
                            e.preventDefault();
                            stop();
                        }}
                        >
                        <StopIcon className="w-3.5 h-3.5" />
                        </Button>
                    )}
                </div>
            </div>
        </form>

        <div className="text-center text-[11px] text-muted-foreground opacity-60">
          <span>Enter to send, Shift + Enter for new line</span>
        </div>
      </div>
    </div>
  );
}