"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { useRouter } from "next/navigation"
import { Button } from "@/components/ui/button"
import { Loader2, Sparkles, Settings2, ArrowUp } from "lucide-react"
import { generateManimVisualization } from "@/lib/api"
import { cn } from "@/lib/utils"
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu"

export default function GenerationForm() {
  const router = useRouter()
  const [prompt, setPrompt] = useState("")
  const [quality, setQuality] = useState("m")
  const [isLoading, setIsLoading] = useState(false)
  const textareaRef = useRef<HTMLTextAreaElement>(null)

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto"
      textareaRef.current.style.height = textareaRef.current.scrollHeight + "px"
    }
  }, [prompt])

  const handleSubmit = async (e?: React.FormEvent) => {
    if (e) e.preventDefault()

    if (!prompt.trim() || isLoading) return

    setIsLoading(true)

    try {
      const response = await generateManimVisualization(prompt, quality)

      // Store job in localStorage
      const jobHistory = JSON.parse(localStorage.getItem("manimJobHistory") || "[]")
      const newJob = {
        id: response.job_id,
        prompt,
        quality,
        timestamp: new Date().toISOString(),
      }

      localStorage.setItem("manimJobHistory", JSON.stringify([newJob, ...jobHistory]))

      // Navigate to job status page
      router.push(`/job/${response.job_id}`)
    } catch (error) {
      console.error("Error generating visualization:", error)
      setIsLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSubmit()
    }
  }

  const qualityLabels = {
    l: "Low Quality (Fast)",
    m: "Medium Quality",
    h: "High Quality (Slow)",
  }

  return (
    <div className="w-full max-w-3xl mx-auto relative group">
      {/* Glow effect behind the input */}
      <div className="absolute -inset-1 bg-gradient-to-r from-primary/20 via-primary/10 to-primary/20 rounded-2xl blur-lg opacity-0 group-hover:opacity-100 transition-opacity duration-500" />

      <div className={cn(
        "relative flex flex-col bg-background/80 backdrop-blur-xl border rounded-2xl transition-all duration-300 shadow-sm",
        "focus-within:shadow-lg focus-within:shadow-primary/5",
        "hover:border-primary/30"
      )}>
        <textarea
          ref={textareaRef}
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Describe a mathematical concept to visualize..."
          className="w-full bg-transparent border-none focus:ring-0 focus:outline-none focus-visible:ring-0 focus-visible:outline-none resize-none p-6 min-h-[120px] text-lg placeholder:text-muted-foreground/50 leading-relaxed"
          disabled={isLoading}
        />

        <div className="flex items-center justify-between px-4 pb-4 mt-2">
          <div className="flex items-center gap-2">
            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <Button variant="ghost" size="sm" className="h-8 gap-2 text-muted-foreground hover:text-primary hover:bg-primary/5 rounded-full px-3 text-xs font-medium border border-transparent hover:border-primary/10 transition-all">
                  <Settings2 className="h-3.5 w-3.5" />
                  {qualityLabels[quality as keyof typeof qualityLabels].split(" ")[0]}
                </Button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start" className="w-48">
                <DropdownMenuItem onClick={() => setQuality("l")} className="gap-2">
                  <span className="w-2 h-2 rounded-full bg-yellow-500/50" />
                  Low (Draft)
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setQuality("m")} className="gap-2">
                  <span className="w-2 h-2 rounded-full bg-blue-500/50" />
                  Medium (Standard)
                </DropdownMenuItem>
                <DropdownMenuItem onClick={() => setQuality("h")} className="gap-2">
                  <span className="w-2 h-2 rounded-full bg-green-500/50" />
                  High (Production)
                </DropdownMenuItem>
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          <Button
            onClick={() => handleSubmit()}
            disabled={!prompt.trim() || isLoading}
            size="icon"
            className={cn(
              "h-10 w-10 rounded-xl transition-all duration-300",
              prompt.trim() ? "bg-primary text-primary-foreground shadow-md shadow-primary/20 hover:scale-105" : "bg-muted text-muted-foreground opacity-50 cursor-not-allowed"
            )}
          >
            {isLoading ? (
              <Loader2 className="h-5 w-5 animate-spin" />
            ) : (
              <ArrowUp className="h-5 w-5" />
            )}
          </Button>
        </div>
      </div>

      <div className="absolute -bottom-6 left-0 right-0 text-center">
        <p className="text-xs text-muted-foreground/60 font-medium">
          Press <kbd className="font-sans px-1 py-0.5 rounded bg-muted border border-border text-[10px]">Enter</kbd> to generate
        </p>
      </div>
    </div>
  )
}
