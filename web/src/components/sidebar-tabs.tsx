"use client";

import React from "react";
import { ChatBubbleIcon, GearIcon, TrashIcon } from "@radix-ui/react-icons";
import * as Tabs from "@radix-ui/react-tabs";
import { Message } from "ai/react";
import Link from "next/link";
import { cn } from "@/lib/utils";
import { ChatOptions } from "./chat/chat-options";
import Settings from "./settings";
import SidebarSkeleton from "./sidebar-skeleton";
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  DialogClose,
} from "./ui/dialog";

interface Chats {
  [key: string]: { chatId: string; messages: Message[] }[];
}

interface SidebarTabsProps {
  isLoading: boolean;
  localChats: Chats;
  selectedChatId: string;
  chatOptions: ChatOptions;
  setChatOptions: React.Dispatch<React.SetStateAction<ChatOptions>>;
  handleDeleteChat: (chatId: string) => void;
}

const SidebarTabs = ({
  localChats,
  selectedChatId,
  isLoading,
  chatOptions,
  setChatOptions,
  handleDeleteChat,
}: SidebarTabsProps) => (
  <Tabs.Root
    className="flex flex-col flex-1 overflow-hidden"
    defaultValue="chats"
  >
    <div className="flex-1 overflow-hidden relative">
      <Tabs.Content className="h-full overflow-y-auto scrollbar-thin scrollbar-thumb-muted" value="chats">
        <div className="pb-4 px-2">
          {isLoading ? (
            <SidebarSkeleton />
          ) : Object.keys(localChats).length > 0 ? (
            Object.keys(localChats).map((group, index) => (
              <div key={index} className="mb-6 mt-4 first:mt-2">
                <h4 className="px-3 mb-2 text-[11px] font-semibold text-muted-foreground/60 uppercase tracking-wider">
                  {group}
                </h4>
                <ul className="space-y-0.5">
                  {localChats[group].map(({ chatId, messages }) => {
                    const isSelected = chatId.substring(5) === selectedChatId;
                    return (
                      <li key={chatId} className="relative group">
                        <Link
                          href={`/chats/${chatId.substring(5)}`}
                          className={cn(
                            "flex items-center gap-2 px-3 py-2 text-sm rounded-md transition-all duration-200",
                            "truncate pr-8",
                            // UPDATED STYLES HERE:
                            isSelected
                              ? "bg-zinc-200/50 dark:bg-white/10 text-foreground font-medium shadow-sm" 
                              : "text-muted-foreground hover:text-foreground hover:bg-muted/50"
                          )}
                        >
                          <span className="truncate">
                            {messages.length > 0 ? messages[0].content : "Empty Chat"}
                          </span>
                        </Link>

                        <div className="absolute right-1 top-1/2 -translate-y-1/2 opacity-0 group-hover:opacity-100 transition-opacity">
                          <Dialog>
                            <DialogTrigger className="p-1.5 rounded-sm hover:bg-background/80 hover:text-red-500 text-muted-foreground transition-colors">
                              <TrashIcon className="w-3.5 h-3.5" />
                            </DialogTrigger>
                            <DialogContent>
                              <DialogHeader className="space-y-3">
                                <DialogTitle>Delete chat?</DialogTitle>
                                <DialogDescription>
                                  Are you sure you want to delete this chat? This action cannot be undone.
                                </DialogDescription>
                                <div className="flex justify-end gap-2 pt-2">
                                  <DialogClose className="px-3 py-2 text-sm rounded-md hover:bg-muted transition-colors">
                                    Cancel
                                  </DialogClose>
                                  <DialogClose
                                    className="px-3 py-2 text-sm rounded-md bg-destructive text-destructive-foreground hover:bg-destructive/90 transition-colors"
                                    onClick={() => handleDeleteChat(chatId)}
                                  >
                                    Delete
                                  </DialogClose>
                                </div>
                              </DialogHeader>
                            </DialogContent>
                          </Dialog>
                        </div>
                      </li>
                    );
                  })}
                </ul>
              </div>
            ))
          ) : (
            <div className="flex flex-col items-center justify-center h-40 text-center px-4">
               <p className="text-sm text-muted-foreground">No chat history</p>
            </div>
          )}
        </div>
      </Tabs.Content>

      <Tabs.Content className="h-full overflow-y-auto p-4" value="settings">
        <Settings chatOptions={chatOptions} setChatOptions={setChatOptions} />
      </Tabs.Content>
    </div>

    <div className="border-t border-border bg-muted/10 p-2">
      <Tabs.List className="grid grid-cols-2 gap-2 bg-muted/20 p-1 rounded-md" aria-label="Sidebar Navigation">
        <Tabs.Trigger
          className={cn(
            "flex items-center justify-center py-1.5 text-sm rounded-sm transition-all",
            "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm",
            "text-muted-foreground hover:text-foreground"
          )}
          value="chats"
        >
          <ChatBubbleIcon className="w-4 h-4 mr-2" />
          <span>Chats</span>
        </Tabs.Trigger>
        <Tabs.Trigger
          className={cn(
            "flex items-center justify-center py-1.5 text-sm rounded-sm transition-all",
            "data-[state=active]:bg-background data-[state=active]:text-foreground data-[state=active]:shadow-sm",
            "text-muted-foreground hover:text-foreground"
          )}
          value="settings"
        >
          <GearIcon className="w-4 h-4 mr-2" />
          <span>Settings</span>
        </Tabs.Trigger>
      </Tabs.List>
    </div>
  </Tabs.Root>
);

export default SidebarTabs;