import React, { useEffect, useRef, useState } from "react";
import Image from "next/image";
import Logo from "../../../public/logo.png"; 
import CodeDisplayBlock from "../code-display-block";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import { Message } from "ai";
import ThinkBlock from "./think-block";
import { cn } from "@/lib/utils";
import { ArrowDownIcon } from "@radix-ui/react-icons";
import { Button } from "../ui/button";

interface ChatListProps {
  messages: Message[];
  isLoading: boolean;
}

const parseMessageContent = (
  content: string,
  isLastMessage: boolean,
  isLoading: boolean
) => {
  let thinkMatch = /<think>([\s\S]*?)(?:<\/think>|$)/i.exec(content);
  if (!thinkMatch && content.includes("</think>")) {
     thinkMatch = /^([\s\S]*?)(?:<\/think>)/i.exec(content);
  }

  let thinkContent = null;
  let mainContent = content;

  if (thinkMatch) {
    thinkContent = thinkMatch[1].trim();
    mainContent = content.replace(thinkMatch[0], "").trim();
    mainContent = mainContent.replace(/<\/think>/gi, "").trim();
  }

  const isThinkingLive =
    isLoading &&
    isLastMessage &&
    (content.includes("<think>") || !content.includes("</think>")) &&
    !content.includes("</think>");

  return { thinkContent, mainContent, isThinkingLive };
};

export default function ChatList({ messages, isLoading }: ChatListProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  
  const [isAtBottom, setIsAtBottom] = useState(true);
  const [showScrollButton, setShowScrollButton] = useState(false);

  const scrollToBottom = () => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
    setIsAtBottom(true);
    setShowScrollButton(false);
  };

  const handleScroll = () => {
    const container = scrollRef.current;
    if (!container) return;

    const { scrollTop, scrollHeight, clientHeight } = container;
    const offset = 100; 
    const isBottom = scrollHeight - scrollTop - clientHeight < offset;

    setIsAtBottom(isBottom);
    setShowScrollButton(!isBottom);
  };

  useEffect(() => {
    if (messages.length > 0) {
      const lastMessage = messages[messages.length - 1];
      
      if (isAtBottom || lastMessage.role === "user") {
        bottomRef.current?.scrollIntoView({ behavior: "auto", block: "end" });
      }
    }
  }, [messages, isAtBottom]);

  if (messages.length === 0) {
    return (
      <div className="w-full h-full flex justify-center items-center p-4">
        <div className="flex flex-col gap-4 items-center opacity-40">
          <Image
            src={Logo}
            alt="AI"
            width={60}
            height={60}
            className="h-16 w-16 object-contain"
          />
          <p className="text-center text-lg text-muted-foreground font-medium">
            How can I help you today?
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="relative w-full h-full overflow-hidden">
      <div
        id="scroller"
        ref={scrollRef}
        onScroll={handleScroll}
        className="w-full h-full overflow-y-auto overflow-x-hidden flex flex-col scroll-smooth"
      >
        <div className="flex-1 py-6 w-full max-w-3xl mx-auto px-4 flex flex-col gap-6">
          {messages
            .filter((message) => message.role !== "system")
            .map((message, index) => {
              const isLastMessage = index === messages.length - 1;
              const { thinkContent, mainContent, isThinkingLive } =
                parseMessageContent(message.content, isLastMessage, isLoading);

              const isUser = message.role === "user";

              return (
                <div
                  key={index}
                  className={cn(
                    "flex w-full",
                    isUser ? "justify-end" : "justify-start"
                  )}
                >
                  <div
                    className={cn(
                      "flex flex-col",
                      // Added min-w-0 to prevent flex item from growing beyond parent
                      isUser ? "items-end max-w-[85%] md:max-w-[75%] min-w-0" : "w-full items-start min-w-0" 
                    )}
                  >
                    <div
                      className={cn(
                        "relative text-[15px] leading-relaxed",
                        isUser
                          ? "px-5 py-3 rounded-2xl rounded-tr-sm bg-blue-600 text-white dark:bg-zinc-800 dark:text-foreground shadow-sm break-words" // Added break-words
                          : "w-full px-0 py-0"
                      )}
                    >
                      {!isUser && thinkContent && (
                        <ThinkBlock
                          content={thinkContent}
                          isLive={isThinkingLive}
                        />
                      )}

                      <div className={cn(!isUser && thinkContent && "mt-4")}>
                        <Markdown
                          remarkPlugins={[remarkGfm, remarkMath]}
                          rehypePlugins={[rehypeKatex]}
                          components={{
                              code({ node, inline, className, children, ...props }: any) {
                              const match = /language-(\w+)/.exec(className || "");
                              const lang = match ? match[1] : "";
                              return !inline && match ? (
                                  <div className="my-6 rounded-md overflow-hidden border border-border">
                                      <CodeDisplayBlock
                                      code={String(children).replace(/\n$/, "")}
                                      lang={lang}
                                      />
                                  </div>
                              ) : (
                                  <code
                                  className={cn(
                                      "px-1.5 py-0.5 rounded-md font-mono text-[13px] border break-all", // break-all for inline code
                                      isUser
                                      ? "bg-white/20 text-white border-transparent" 
                                      : "bg-muted text-foreground border-border"
                                  )}
                                  {...props}
                                  >
                                  {children}
                                  </code>
                              );
                              },
                              h1: ({children}) => <h1 className="text-3xl font-bold mt-8 mb-4 break-words">{children}</h1>,
                              h2: ({children}) => <h2 className="text-2xl font-semibold mt-8 mb-4 border-b pb-2 break-words">{children}</h2>,
                              h3: ({children}) => <h3 className="text-xl font-semibold mt-6 mb-3 break-words">{children}</h3>,
                              h4: ({children}) => <h4 className="text-lg font-semibold mt-6 mb-3 break-words">{children}</h4>,
                              
                              // Paragraphs: whitespace-pre-wrap ensures formatting is kept, break-words handles long strings
                              p: ({children}) => <p className="mb-5 last:mb-0 leading-7 whitespace-pre-wrap break-words">{children}</p>,
                              
                              a: ({children, ...props}: any) => (
                                  <a className="text-blue-500 hover:underline cursor-pointer font-medium break-all" {...props}>{children}</a>
                              ),
                              ul: ({children}) => <ul className="list-disc pl-6 mb-5 space-y-2">{children}</ul>,
                              ol: ({children}) => <ol className="list-decimal pl-6 mb-5 space-y-2">{children}</ol>,
                              li: ({children}) => <li className="pl-1 leading-7">{children}</li>,
                              blockquote: ({children}) => (
                                  <blockquote className="border-l-4 border-primary/20 bg-muted/40 pl-4 py-2 my-4 rounded-r-md italic">
                                      {children}
                                  </blockquote>
                              ),
                              table: ({children}) => <div className="overflow-x-auto my-6 border rounded-md"><table className="w-full text-sm text-left">{children}</table></div>,
                              th: ({children}) => <th className="bg-muted px-4 py-3 border-b font-semibold">{children}</th>,
                              td: ({children}) => <td className="px-4 py-3 border-b last:border-0">{children}</td>,
                          }}
                        >
                          {isUser ? message.content : mainContent}
                        </Markdown>
                      </div>

                      {isLoading &&
                        isLastMessage &&
                        !isUser &&
                        !isThinkingLive &&
                        mainContent.length === 0 && (
                          <div className="flex items-center gap-1 h-6 mt-2 opacity-50">
                            <span className="w-1.5 h-1.5 bg-foreground rounded-full animate-bounce [animation-delay:-0.3s]"></span>
                            <span className="w-1.5 h-1.5 bg-foreground rounded-full animate-bounce [animation-delay:-0.15s]"></span>
                            <span className="w-1.5 h-1.5 bg-foreground rounded-full animate-bounce"></span>
                          </div>
                        )}
                    </div>
                  </div>
                </div>
              );
            })}
        </div>
        <div id="anchor" ref={bottomRef} className="h-1"></div>
      </div>

      {showScrollButton && (
        <Button
          variant="outline"
          size="icon"
          className={cn(
            "absolute bottom-4 right-1/2 translate-x-1/2 rounded-full shadow-lg z-50 w-10 h-10 animate-in fade-in zoom-in duration-300",
            "!bg-blue-600 hover:!bg-blue-700 !text-white !border-0"
          )}
          onClick={scrollToBottom}
        >
          <ArrowDownIcon className="w-5 h-5" />
        </Button>
      )}
    </div>
  );
}