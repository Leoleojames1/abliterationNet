"use client"

import Link from "next/link"
import { Card } from "@/components/ui/card"
import { ThemeProvider } from "@/components/theme-provider"
import { Brain, Atom, Wand2, User } from 'lucide-react'
import { Button } from "@/components/ui/button"

export default function Home() {
  const tools = [
    {
      title: "Standard Abliterator",
      description: "Interactive neural network ablation visualization and analysis tool",
      href: "/abliterator",
      Icon: Brain
    },
    {
      title: "Berezinian Ablator",
      description: "Explore neural networks through the lens of Berezinian algebra",
      href: "/berezinianAblator",
      Icon: Atom
    },
    {
      title: "CVLI Ablator",
      description: "Contour Vector Line Integral Ablation visualization tool",
      href: "/CVLIAblator",
      Icon: Wand2
    },
  ]

  return (
    <ThemeProvider attribute="class" defaultTheme="dark" enableSystem>
      <div className="container mx-auto p-4 min-h-screen">
        <header className="flex justify-between items-center mb-8">
          <h1 className="text-3xl font-bold">Abliteration Net</h1>
          <Button variant="ghost" size="icon">
            <User className="h-6 w-6" />
          </Button>
        </header>

        <div className="text-center mb-12">
          <p className="text-xl opacity-70">
            Welcome to the comprehensive suite of abliteration and visualization tools for transformer neural network ablation & analysis.
          </p>
        </div>

        <div className="grid md:grid-cols-2 lg:grid-cols-3 gap-6">
          {tools.map((tool) => (
            <Link href={tool.href} key={tool.href} className="block">
              <Card className="h-full p-6 hover:bg-muted/50 transition-colors">
                <div className="mb-4">
                  <tool.Icon className="w-12 h-12" />
                </div>
                <h2 className="text-2xl font-semibold mb-2">{tool.title}</h2>
                <p className="text-muted-foreground">{tool.description}</p>
              </Card>
            </Link>
          ))}
        </div>

        <footer className="mt-12 text-center text-muted-foreground">
          <p>&copy; 2024 Abliteration Net. All rights reserved.</p>
        </footer>
      </div>
    </ThemeProvider>
  )
}

