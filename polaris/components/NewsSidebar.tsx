"use client";

import React, { useState } from "react";
import { Sidebar, SidebarBody, SidebarLink } from "./ui/sidebar";
import {
  IconHome,
  IconSearch,
  IconNews,
  IconTrendingUp,
  IconShield,
  IconSettings,
  IconBookmark,
} from "@tabler/icons-react";
import { motion } from "motion/react";
import { cn } from "@/lib/utils";
import { Newspaper } from "lucide-react";
import Link from "next/link";

export function NewsSidebar({ children }: { children: React.ReactNode }) {
  const links = [
    {
      label: "Home",
      href: "/",
      icon: (
        <IconHome className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
      ),
    },
    {
      label: "Search Articles",
      href: "/search",
      icon: (
        <IconSearch className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
      ),
    },
    {
      label: "Latest News",
      href: "/",
      icon: (
        <IconNews className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
      ),
    },
    {
      label: "Trending",
      href: "#trending",
      icon: (
        <IconTrendingUp className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
      ),
    },
    // {
    //   label: "Bias Analysis",
    //   href: "#bias",
    //   icon: (
    //     <IconShield className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
    //   ),
    // },
    {
      label: "Saved Articles",
      href: "#saved",
      icon: (
        <IconBookmark className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
      ),
    },
    {
      label: "Settings",
      href: "#settings",
      icon: (
        <IconSettings className="h-5 w-5 shrink-0 text-gray-400 group-hover/sidebar:text-blue-400 transition-colors duration-200" />
      ),
    },
  ];

  const [open, setOpen] = useState(false);

  return (
    <div
      className={cn(
        "flex w-full flex-col md:flex-row min-h-screen"
      )}
    >
      <div className="bg-zinc-900 md:h-screen md:sticky md:top-0">
        <Sidebar open={open} setOpen={setOpen}>
          <SidebarBody className="justify-between h-screen">
            <div className="flex flex-1 flex-col overflow-x-hidden overflow-y-auto">
              {open ? <Logo /> : <LogoIcon />}
              <div className="mt-8 flex flex-col gap-2">
                {links.map((link, idx) => (
                  <SidebarLink key={idx} link={link} />
                ))}
              </div>
            </div>
            <div className="shrink-0 pt-4 border-t border-zinc-800">
              <SidebarLink
                link={{
                  label: "Polaris User",
                  href: "#profile",
                  icon: (
                    <div className="h-7 w-7 shrink-0 rounded-full bg-gradient-to-r from-blue-500 to-cyan-500 flex items-center justify-center text-white font-bold text-sm shadow-md">
                      P
                    </div>
                  ),
                }}
              />
            </div>
          </SidebarBody>
        </Sidebar>
      </div>
      <Dashboard>{children}</Dashboard>
    </div>
  );
}

export const Logo = () => {
  return (
    <Link
      href="/"
      className="relative z-20 flex items-center space-x-2 py-1 text-sm font-normal hover:opacity-80 transition-opacity"
    >
      <div className="h-7 w-7 shrink-0 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
        <Newspaper className="w-4 h-4 text-white" />
      </div>
      <motion.span
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        className="font-bold whitespace-pre text-gray-100"
      >
        Polaris News
      </motion.span>
    </Link>
  );
};

export const LogoIcon = () => {
  return (
    <Link
      href="/"
      className="relative z-20 flex items-center justify-center py-1 text-sm font-normal hover:opacity-80 transition-opacity"
    >
      <div className="h-7 w-7 shrink-0 rounded-lg bg-blue-600 flex items-center justify-center shadow-lg shadow-blue-500/20">
        <Newspaper className="w-4 h-4 text-white" />
      </div>
    </Link>
  );
};

// Dashboard wrapper component
const Dashboard = ({ children }: { children: React.ReactNode }) => {
  return (
    <div className="flex flex-1 overflow-hidden">
      <div className="flex h-full w-full flex-1 flex-col bg-black overflow-y-auto">
        {children}
      </div>
    </div>
  );
};

