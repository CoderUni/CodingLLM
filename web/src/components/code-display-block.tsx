"use client";

import React from "react";
import { CheckIcon, CopyIcon } from "@radix-ui/react-icons";
import { useTheme } from "next-themes";
import { toast } from "sonner";
import { Button } from "./ui/button";
import { Prism as SyntaxHighlighter } from "react-syntax-highlighter";
import { oneDark, oneLight } from "react-syntax-highlighter/dist/cjs/styles/prism";

interface CodeDisplayBlockProps {
  code: string;
  lang: string;
}

export default function CodeDisplayBlock({ code, lang }: CodeDisplayBlockProps) {
  const [isCopied, setisCopied] = React.useState(false);
  const { theme } = useTheme();

  const copyToClipboard = () => {
    navigator.clipboard.writeText(code);
    setisCopied(true);
    toast.success("Code copied to clipboard!");
    setTimeout(() => {
      setisCopied(false);
    }, 1500);
  };

  return (
    <div className="relative my-4 flex flex-col text-start border rounded-md overflow-hidden border-border">
      {/* Header */}
      <div className="flex items-center justify-between bg-muted/50 px-4 py-2 border-b border-border">
        <span className="text-xs font-medium text-muted-foreground lowercase">
          {lang || "text"}
        </span>
        <Button
          onClick={copyToClipboard}
          variant="ghost"
          size="icon"
          className="h-7 w-7 hover:bg-background focus-visible:ring-1 focus-visible:ring-ring"
          aria-label="Copy code"
        >
          {isCopied ? (
            <CheckIcon className="w-4 h-4 text-green-500 scale-110 transition-all" />
          ) : (
            <CopyIcon className="w-4 h-4 text-muted-foreground scale-100 transition-all" />
          )}
        </Button>
      </div>
      
      {/* Code Block */}
      <div className="overflow-x-auto">
        <SyntaxHighlighter
          language={lang}
          style={theme === "dark" ? oneDark : oneLight}
          preload={false}
          // Custom styles to match Shadcn UI aesthetics
          customStyle={{
            margin: 0,
            padding: "1.5rem",
            background: theme === "dark" ? "#101012" : "#fcfcfc", // Matches standard dark/light backgrounds
            fontSize: "0.875rem",
            lineHeight: "1.5",
          }}
          codeTagProps={{
            style: {
              fontSize: "0.9rem",
              fontFamily: "var(--font-mono)",
            },
          }}
        >
          {code}
        </SyntaxHighlighter>
      </div>
    </div>
  );
}