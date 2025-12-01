"use client";

import React, { useEffect, useState } from "react";
import { useTheme } from "next-themes";
import { ChatOptions } from "./chat/chat-options";
import { Slider } from "./ui/slider";
import { Button } from "./ui/button";
import ClearChatsButton from "./settings-clear-chats";
import SystemPrompt from "./system-prompt";
import { 
  Sun, 
  Moon, 
  Thermometer, 
  Cpu, 
  Activity, 
  Hash 
} from "lucide-react";
import { useHasMounted } from "@/lib/utils";

interface SettingsProps {
  chatOptions: ChatOptions;
  setChatOptions: React.Dispatch<React.SetStateAction<ChatOptions>>;
}

export default function Settings({ chatOptions, setChatOptions }: SettingsProps) {
  const hasMounted = useHasMounted();
  const { setTheme, theme } = useTheme();

  if (!hasMounted) return null;

  const handleSliderChange = (key: keyof ChatOptions, value: number[]) => {
    setChatOptions((prev) => ({ ...prev, [key]: value[0] }));
  };

  const nextTheme = theme === "light" ? "dark" : "light";

  return (
    <div className="flex flex-col gap-8 p-4 w-full h-full min-h-full overflow-y-auto pb-6">
      
      {/* 1. System Instructions */}
      <SystemPrompt chatOptions={chatOptions} setChatOptions={setChatOptions} />

      {/* 2. Model Parameters */}
      <div className="space-y-6">
        <label className="text-xs font-bold text-muted-foreground uppercase tracking-widest px-1">
           Model Parameters
        </label>

        {/* Temperature */}
        <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-2">
                    <Thermometer className="w-4 h-4 stroke-[2.5] text-blue-500" />
                    <span className="font-medium">Temperature</span>
                </div>
                <span className="text-muted-foreground font-mono text-xs bg-muted px-2 py-0.5 rounded">
                    {chatOptions.temperature ?? 0.6}
                </span>
            </div>
            <Slider
                defaultValue={[chatOptions.temperature ?? 0.6]}
                max={1}
                min={0}
                step={0.1}
                onValueChange={(val) => handleSliderChange("temperature", val)}
            />
        </div>

        {/* Top P */}
        <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-2">
                    <Activity className="w-4 h-4 stroke-[2.5] text-purple-500" />
                    <span className="font-medium">Top P</span>
                </div>
                <span className="text-muted-foreground font-mono text-xs bg-muted px-2 py-0.5 rounded">
                    {chatOptions.topP ?? 0.95}
                </span>
            </div>
            <Slider
                defaultValue={[chatOptions.topP ?? 0.95]}
                max={1}
                min={0}
                step={0.01}
                onValueChange={(val) => handleSliderChange("topP", val)}
            />
        </div>

         {/* Top K */}
         <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-2">
                    <Hash className="w-4 h-4 stroke-[2.5] text-green-500" />
                    <span className="font-medium">Top K</span>
                </div>
                <span className="text-muted-foreground font-mono text-xs bg-muted px-2 py-0.5 rounded">
                    {chatOptions.topK ?? 20}
                </span>
            </div>
            <Slider
                defaultValue={[chatOptions.topK ?? 20]}
                max={100}
                min={0}
                step={1}
                onValueChange={(val) => handleSliderChange("topK", val)}
            />
        </div>

        {/* Min P */}
        <div className="space-y-3">
            <div className="flex justify-between items-center text-sm">
                <div className="flex items-center gap-2">
                    <Cpu className="w-4 h-4 stroke-[2.5] text-orange-500" />
                    <span className="font-medium">Min P</span>
                </div>
                <span className="text-muted-foreground font-mono text-xs bg-muted px-2 py-0.5 rounded">
                    {chatOptions.minP ?? 0.0}
                </span>
            </div>
            <Slider
                defaultValue={[chatOptions.minP ?? 0.0]}
                max={1}
                min={0}
                step={0.01}
                onValueChange={(val) => handleSliderChange("minP", val)}
            />
        </div>
      </div>

      {/* Spacer / Bottom Section */}
      <div className="mt-auto pt-10 flex flex-col gap-4 border-t border-border/40">
        
        {/* Theme Toggle */}
        <Button
          className="justify-start gap-4 w-full h-12 font-medium border bg-background hover:bg-accent/50 rounded-lg px-4"
          variant="ghost"
          onClick={() => setTheme(nextTheme)}
        >
          {nextTheme === "light" ? (
            <Sun className="w-5 h-5 stroke-[2.5] text-orange-500 shrink-0" />
          ) : (
            <Moon className="w-5 h-5 stroke-[2.5] text-blue-500 shrink-0" />
          )}
          {/* Responsive Text Size */}
          <span className="flex-1 text-left text-xs sm:text-sm truncate whitespace-nowrap">
            {nextTheme === "light" ? "Light mode" : "Dark mode"}
          </span>
        </Button>

        {/* Clear Chats */}
        <ClearChatsButton />
      </div>

    </div>
  );
}