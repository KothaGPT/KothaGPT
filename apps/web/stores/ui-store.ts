"use client";

import { create } from "zustand";

interface UIState {
  commandOpen: boolean;
  sidebarCollapsed: boolean;
  setCommandOpen: (open: boolean) => void;
  toggleSidebar: () => void;
}

export const useUIStore = create<UIState>((set) => ({
  commandOpen: false,
  sidebarCollapsed: false,
  setCommandOpen: (open) => set({ commandOpen: open }),
  toggleSidebar: () => set((s) => ({ sidebarCollapsed: !s.sidebarCollapsed })),
}));
