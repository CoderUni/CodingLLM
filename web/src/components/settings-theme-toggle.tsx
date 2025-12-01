"use client";

import * as React from "react";
import { MoonIcon, SunIcon } from "@radix-ui/react-icons";
import { useTheme } from "next-themes";
import { useHasMounted } from "@/lib/utils";
import { Button } from "./ui/button";

export default function SettingsThemeToggle() {
  const hasMounted = useHasMounted();
  const { setTheme, theme } = useTheme();

  if (!hasMounted) return null;

  const nextTheme = theme === "light" ? "dark" : "light";

  return (
    <Button
      className="justify-start gap-3 w-full border-none shadow-none hover:bg-muted/50"
      variant="outline"
      onClick={() => setTheme(nextTheme)}
    >
      {nextTheme === "light" ? (
        <SunIcon className="w-4 h-4 text-orange-500" />
      ) : (
        <MoonIcon className="w-4 h-4 text-blue-500" />
      )}
      <span className="font-normal">{nextTheme === "light" ? "Switch to Light mode" : "Switch to Dark mode"}</span>
    </Button>
  );
}