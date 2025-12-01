"use client";
import { useEffect, useState } from "react";
import { Edit2 } from 'lucide-react';
import { Message } from "ai/react";
import Image from "next/image";
import Link from "next/link";
import Logo from "../../public/logo.png";
import { ChatOptions } from "./chat/chat-options";
import SidebarTabs from "./sidebar-tabs";
import { cn } from "@/lib/utils";

interface SidebarProps {
  isCollapsed: boolean;
  onClick?: () => void;
  isMobile: boolean;
  chatId: string;
  setChatId: React.Dispatch<React.SetStateAction<string>>;
  chatOptions: ChatOptions;
  setChatOptions: React.Dispatch<React.SetStateAction<ChatOptions>>;
}

interface Chats {
  [key: string]: { chatId: string; messages: Message[] }[];
}

export function Sidebar({
  isCollapsed,
  isMobile,
  chatId,
  setChatId,
  chatOptions,
  setChatOptions,
}: SidebarProps) {
  const [localChats, setLocalChats] = useState<Chats>({});
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    setLocalChats(getLocalstorageChats());
    const handleStorageChange = () => {
      setLocalChats(getLocalstorageChats());
    };
    window.addEventListener("storage", handleStorageChange);
    return () => {
      window.removeEventListener("storage", handleStorageChange);
    };
  }, [chatId]);

  const getLocalstorageChats = (): Chats => {
    const chats = Object.keys(localStorage).filter((key) =>
      key.startsWith("chat_")
    );

    if (chats.length === 0) {
      setIsLoading(false);
    }

    const chatObjects = chats.map((chat) => {
      const item = localStorage.getItem(chat);
      return item
        ? { chatId: chat, messages: JSON.parse(item) }
        : { chatId: "", messages: [] };
    });

    chatObjects.sort((a, b) => {
      const aDate = new Date(a.messages[0].createdAt);
      const bDate = new Date(b.messages[0].createdAt);
      return bDate.getTime() - aDate.getTime();
    });

    const groupChatsByDate = (
      chats: { chatId: string; messages: Message[] }[]
    ) => {
      const today = new Date();
      const groupedChats: Chats = {};

      chats.forEach((chat) => {
        const createdAt = new Date(chat.messages[0].createdAt ?? "");
        const diffInDays = Math.floor(
          (today.getTime() - createdAt.getTime()) / (1000 * 3600 * 24)
        );

        let group: string;
        if (diffInDays === 0) group = "Today";
        else if (diffInDays === 1) group = "Yesterday";
        else if (diffInDays <= 7) group = "Previous 7 Days";
        else if (diffInDays <= 30) group = "Previous 30 Days";
        else group = "Older";

        if (!groupedChats[group]) {
          groupedChats[group] = [];
        }
        groupedChats[group].push(chat);
      });

      return groupedChats;
    };

    setIsLoading(false);
    return groupChatsByDate(chatObjects);
  };

  const handleDeleteChat = (chatId: string) => {
    localStorage.removeItem(chatId);
    setLocalChats(getLocalstorageChats());
  };

  return (
    <div
      data-collapsed={isCollapsed}
      className="relative group bg-muted/10 border-r border-border flex flex-col h-full data-[collapsed=true]:p-0 data-[collapsed=true]:hidden transition-all duration-300 w-[260px]"
    >
      {/* Header / New Chat */}
      <div className="flex flex-col p-3 gap-2 sticky top-0 bg-background/95 backdrop-blur z-20 border-b border-border/50">
        <Link
          href="/"
          onClick={() => setChatId("")}
          className={cn(
            "flex items-center justify-between px-3 py-2.5 rounded-md text-sm font-medium transition-all duration-200",
            "bg-primary text-primary-foreground hover:bg-primary/90 shadow-sm", // Primary Button Style
            "border border-transparent"
          )}
        >
          <div className="flex items-center gap-2">
            {!isCollapsed && !isMobile && (
              <Image
                src={Logo}
                alt="AI"
                width={16}
                height={16}
                className="opacity-90"
              />
            )}
            <span>New Chat</span>
          </div>
          <Edit2 strokeWidth={2.5} className="w-4 h-4 opacity-80" />
        </Link>
      </div>

      {/* Tabs / Content */}
      <SidebarTabs
        isLoading={isLoading}
        localChats={localChats}
        selectedChatId={chatId}
        chatOptions={chatOptions}
        setChatOptions={setChatOptions}
        handleDeleteChat={handleDeleteChat}
      />
    </div>
  );
}