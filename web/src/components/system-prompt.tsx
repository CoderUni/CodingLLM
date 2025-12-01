"use client";

import { Dispatch, SetStateAction, useEffect, useState } from "react";
import { toast } from "sonner";
import { useDebounce } from "use-debounce";
import { useHasMounted } from "@/lib/utils";
import { ChatOptions } from "./chat/chat-options";
import { Textarea } from "./ui/textarea";

export interface SystemPromptProps {
  chatOptions: ChatOptions;
  setChatOptions: Dispatch<SetStateAction<ChatOptions>>;
}

export default function SystemPrompt({
  chatOptions,
  setChatOptions,
}: SystemPromptProps) {
  const hasMounted = useHasMounted();
  const systemPrompt = chatOptions ? chatOptions.systemPrompt : "";
  const [text, setText] = useState<string>(systemPrompt || "");
  const [debouncedText] = useDebounce(text, 500);

  useEffect(() => {
    if (!hasMounted) return;
    if (debouncedText !== systemPrompt) {
      setChatOptions({ ...chatOptions, systemPrompt: debouncedText });
      toast.success("System prompt updated");
    }
  }, [hasMounted, debouncedText]);

  return (
    <div className="flex flex-col gap-3">
      <label className="text-xs font-bold text-muted-foreground uppercase tracking-widest px-1">
        System Instructions
      </label>
      <Textarea
        className="rounded-sm resize-none min-h-[150px] text-sm bg-muted/30 focus:bg-background transition-colors border-input/50 focus:border-primary/50"
        value={text}
        onChange={(e) => setText(e.currentTarget.value)}
        placeholder="You are a helpful assistant."
      />
    </div>
  );
}