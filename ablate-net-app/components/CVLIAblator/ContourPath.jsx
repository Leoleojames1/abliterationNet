import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

export function ContourPath({ points, color = 0x00ff00 }) {
  const lineRef = useRef()

  const geometry = useMemo(() => {
    const geometry = new THREE.BufferGeometry().setFromPoints(points.map(p => new THREE.Vector3(p[0], p[1], p[2])))
    return geometry
  }, [points])

  useFrame(() => {
    if (lineRef.current) {
      lineRef.current.rotation.y += 0.002
    }
  })

  return (
    <line ref={lineRef}>
      <bufferGeometry attach="geometry" {...geometry} />
      <lineBasicMaterial attach="material" color={color} linewidth={2} />
    </line>
  )
}

