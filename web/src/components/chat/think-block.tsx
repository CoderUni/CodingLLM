"use client";

import React, { useState } from "react";
import { ChevronDownIcon, ChevronRightIcon } from "@radix-ui/react-icons";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { cn } from "@/lib/utils";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";

interface ThinkBlockProps {
    content: string;
    isLive?: boolean;
}

export default function ThinkBlock({ content, isLive = false }: ThinkBlockProps) {
    const [isExpanded, setIsExpanded] = useState(isLive);

    if (!content) return null;

    return (
        <div className="my-4 group">
            <button
                onClick={() => setIsExpanded(!isExpanded)}
                className={cn(
                    "flex items-center gap-2 text-xs font-semibold uppercase tracking-wider text-muted-foreground hover:text-foreground transition-colors mb-2 select-none",
                    isLive && "animate-pulse"
                )}
                aria-expanded={isExpanded}
            >
                {isExpanded ? (
                    <ChevronDownIcon className="w-3.5 h-3.5" />
                ) : (
                    <ChevronRightIcon className="w-3.5 h-3.5" />
                )}
                <span>{isLive ? "Reasoning..." : "Thought Process"}</span>
            </button>

            {isExpanded && (
                <div className="relative pl-4 border-l-2 border-primary/20 hover:border-primary/40 transition-colors">
                    {/* Readability Improvements:
              1. text-sm: Matches main text size (was text-[13px])
              2. text-zinc-600: Darker/Higher contrast than muted-foreground
            */}
                    <div className="text-sm text-zinc-600 dark:text-zinc-300 leading-relaxed animate-in fade-in slide-in-from-top-1 duration-200">
                        <Markdown
                            remarkPlugins={[remarkGfm, remarkMath]}
                            rehypePlugins={[rehypeKatex]}
                            components={{
                                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>
                            }}
                        >
                            {content}
                        </Markdown>
                        {isLive && (
                            <span className="inline-block w-1.5 h-3 ml-1 bg-primary/50 animate-pulse align-middle" />
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}