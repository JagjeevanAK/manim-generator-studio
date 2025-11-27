"use client"

import GenerationForm from "@/components/GenerateForm"
import { Accordion, AccordionContent, AccordionItem, AccordionTrigger } from "@/components/ui/accordion"
import { Sparkles, Zap, Palette, Download, Code2, LineChart, Cpu, Lightbulb, ArrowRight } from "lucide-react"

export default function Home() {
  return (
    <div className="min-h-screen">
      {/* Hero Section */}
      <section className="relative pt-32 pb-20 px-4 overflow-hidden">
        <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-primary/10 via-background to-background"></div>

        <div className="max-w-5xl mx-auto text-center space-y-8">
          <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-primary/5 border border-primary/10 text-primary text-sm font-medium animate-fade-in">
            <Sparkles className="h-3.5 w-3.5" />
            <span>AI-Powered Animation Generator</span>
          </div>

          <h1 className="text-6xl md:text-7xl font-bold tracking-tight text-foreground animate-fade-in [animation-delay:100ms]">
            Mathematical Animations
            <span className="block mt-2 pb-4 bg-clip-text text-transparent bg-gradient-to-r from-primary via-primary/80 to-primary/50">
              Reimagined with AI
            </span>
          </h1>

          <p className="text-xl text-muted-foreground max-w-2xl mx-auto leading-relaxed animate-fade-in [animation-delay:200ms]">
            Transform your ideas into precise Manim visualizations.
            Simply describe the math, and watch it come to life.
          </p>

          <div className="max-w-3xl mx-auto mt-12 animate-fade-in [animation-delay:300ms]">
            <GenerationForm />
          </div>
        </div>
      </section>

      {/* Features Grid */}
      <section className="py-24 px-4">
        <div className="max-w-5xl mx-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-x-12 gap-y-16">
            <div className="space-y-4 group">
              <div className="h-12 w-12 rounded-2xl bg-primary/5 flex items-center justify-center group-hover:bg-primary/10 transition-colors duration-300">
                <Sparkles className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">AI-Powered</h3>
              <p className="text-muted-foreground leading-relaxed">
                Advanced models translate natural language into precise Manim Python code instantly.
              </p>
            </div>

            <div className="space-y-4 group">
              <div className="h-12 w-12 rounded-2xl bg-primary/5 flex items-center justify-center group-hover:bg-primary/10 transition-colors duration-300">
                <Zap className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Fast Rendering</h3>
              <p className="text-muted-foreground leading-relaxed">
                Optimized rendering pipeline delivers your animations in seconds, not hours.
              </p>
            </div>

            <div className="space-y-4 group">
              <div className="h-12 w-12 rounded-2xl bg-primary/5 flex items-center justify-center group-hover:bg-primary/10 transition-colors duration-300">
                <Palette className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Beautiful Output</h3>
              <p className="text-muted-foreground leading-relaxed">
                Studio-quality mathematical visualizations ready for your next presentation.
              </p>
            </div>

            <div className="space-y-4 group">
              <div className="h-12 w-12 rounded-2xl bg-primary/5 flex items-center justify-center group-hover:bg-primary/10 transition-colors duration-300">
                <Download className="h-6 w-6 text-primary" />
              </div>
              <h3 className="text-xl font-semibold">Full Control</h3>
              <p className="text-muted-foreground leading-relaxed">
                Download the video or grab the generated Python code to tweak it further.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Examples Section */}
      <section className="py-24 px-4 bg-secondary/30">
        <div className="max-w-6xl mx-auto">
          <div className="flex flex-col md:flex-row justify-between items-end mb-16 gap-6">
            <div className="space-y-2">
              <h2 className="text-4xl font-bold tracking-tight">Limitless Possibilities</h2>
              <p className="text-muted-foreground text-lg">From basic geometry to complex physics simulations.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
            {[
              { icon: LineChart, title: "Calculus", desc: "Visualize limits, derivatives, and integrals dynamically." },
              { icon: Cpu, title: "Algorithms", desc: "Animate sorting, searching, and graph traversals." },
              { icon: Code2, title: "Physics", desc: "Simulate forces, motion, and particle systems." },
              { icon: Lightbulb, title: "Education", desc: "Create engaging explainers for complex topics." },
            ].map((item, i) => (
              <div key={i} className="p-6 rounded-3xl bg-background border border-border/50 hover:border-primary/20 transition-all duration-300 hover:shadow-lg hover:shadow-primary/5">
                <item.icon className="h-8 w-8 text-primary mb-4" />
                <h3 className="text-lg font-semibold mb-2">{item.title}</h3>
                <p className="text-sm text-muted-foreground">{item.desc}</p>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-16">How It Works</h2>

          <div className="relative">
            {/* Connecting Line */}
            <div className="absolute left-[27px] top-8 bottom-8 w-0.5 bg-gradient-to-b from-primary/20 via-primary/20 to-transparent hidden md:block"></div>

            <div className="space-y-12">
              {[
                { step: "01", title: "Describe Your Idea", desc: "Type your visualization concept in plain English. Be as specific or abstract as you like." },
                { step: "02", title: "AI Generation", desc: "Our engine translates your words into precise Manim Python code, optimizing for visual clarity." },
                { step: "03", title: "Render & Download", desc: "Watch your animation come to life in seconds. Download the video or the source code." },
              ].map((item, i) => (
                <div key={i} className="relative flex gap-8 items-start">
                  <div className="flex-shrink-0 h-14 w-14 rounded-full bg-background border-2 border-primary/20 flex items-center justify-center z-10 font-bold text-primary text-lg shadow-sm">
                    {item.step}
                  </div>
                  <div className="pt-2">
                    <h3 className="text-xl font-semibold mb-2">{item.title}</h3>
                    <p className="text-muted-foreground leading-relaxed max-w-xl">{item.desc}</p>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </section>

      {/* FAQ Section */}
      <section className="py-24 px-4 bg-secondary/30">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-3xl font-bold text-center mb-12">Common Questions</h2>
          <Accordion type="single" collapsible className="w-full space-y-4">
            {[
              { q: "What is Manim Generator Studio?", a: "An AI-powered tool that generates mathematical animations using the Manim library from natural language descriptions." },
              { q: "Do I need to know Python?", a: "Not at all. You describe what you want, and we handle the code. You can view the code if you're interested in learning, though!" },
              { q: "Is it free to use?", a: "Yes, currently it is free to use. We may introduce premium features for higher quality or longer renders in the future." },
              { q: "Can I use the videos commercially?", a: "Yes, the videos are yours. Just make sure to comply with the MIT license of the Manim library." },
            ].map((item, i) => (
              <AccordionItem key={i} value={`item-${i}`} className="border-none bg-background rounded-2xl px-6 shadow-sm">
                <AccordionTrigger className="hover:text-primary hover:no-underline py-6 text-left font-medium">
                  {item.q}
                </AccordionTrigger>
                <AccordionContent className="text-muted-foreground pb-6 leading-relaxed">
                  {item.a}
                </AccordionContent>
              </AccordionItem>
            ))}
          </Accordion>
        </div>
      </section>

      {/* Bottom CTA */}
      <section className="py-24 px-4">
        <div className="max-w-4xl mx-auto text-center">
          <h2 className="text-4xl md:text-5xl font-bold mb-6 tracking-tight">
            Ready to visualize?
          </h2>
          <p className="text-xl text-muted-foreground mb-10 max-w-2xl mx-auto">
            Join thousands of students, teachers, and creators making beautiful math animations.
          </p>
          <button
            onClick={() => window.scrollTo({ top: 0, behavior: 'smooth' })}
            className="group inline-flex items-center gap-2 bg-primary text-primary-foreground px-8 py-4 rounded-full font-medium text-lg hover:bg-primary/90 transition-all hover:scale-105 shadow-lg shadow-primary/20"
          >
            <Sparkles className="h-5 w-5" />
            Start Creating
            <ArrowRight className="h-5 w-5 group-hover:translate-x-1 transition-transform" />
          </button>
        </div>
      </section>
    </div>
  )
}
