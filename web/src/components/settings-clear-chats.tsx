"use client";

import * as React from "react";
import {
  Dialog,
  DialogClose,
  DialogContent,
  DialogDescription,
  DialogTrigger,
  DialogHeader,
  DialogTitle
} from "./ui/dialog";
import { Trash2 } from "lucide-react";
import { useRouter } from "next/navigation";
import { useHasMounted } from "@/lib/utils";
import { Button } from "./ui/button";

export default function ClearChatsButton() {
  const hasMounted = useHasMounted();
  const router = useRouter();

  if (!hasMounted) return null;

  const chats = Object.keys(localStorage).filter((key) =>
    key.startsWith("chat_")
  );

  const disabled = chats.length === 0;

  const clearChats = () => {
    chats.forEach((key) => localStorage.removeItem(key));
    window.dispatchEvent(new Event("storage"));
    router.push("/");
  };

  return (
    <Dialog>
      <DialogTrigger asChild>
        <Button 
            variant="ghost" 
            className="w-full justify-start gap-4 h-12 px-4 text-red-500 hover:text-red-600 hover:bg-red-50 dark:hover:bg-red-900/10 transition-colors rounded-lg"
            disabled={disabled}
        >
            {/* Icon: shrink-0 prevents it from getting squished */}
            <Trash2 className="w-5 h-5 stroke-[2.5] shrink-0" />
            {/* Responsive Text Size */}
            <span className="flex-1 text-left font-normal text-xs sm:text-sm truncate whitespace-nowrap">
                Clear all chats
            </span>
        </Button>
      </DialogTrigger>
      <DialogContent>
        <DialogHeader>
          <DialogTitle>Clear History</DialogTitle>
          <DialogDescription>
            Are you sure you want to delete all chat history? This action cannot be undone.
          </DialogDescription>
        </DialogHeader>
        <div className="flex justify-end gap-2 mt-4">
            <DialogClose asChild>
                <Button variant="outline" size="sm">Cancel</Button>
            </DialogClose>
            <DialogClose asChild>
                <Button variant="destructive" size="sm" onClick={clearChats}>Delete All</Button>
            </DialogClose>
        </div>
      </DialogContent>
    </Dialog>
  );
}