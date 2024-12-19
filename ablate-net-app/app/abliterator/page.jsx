"use client"

import AbliteratorController from '@/components/abliterator/AbliteratorController'
import { ThemeProvider } from "@/components/theme-provider"

export default function Home() {
  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <AbliteratorController />
    </ThemeProvider>
  )
}

